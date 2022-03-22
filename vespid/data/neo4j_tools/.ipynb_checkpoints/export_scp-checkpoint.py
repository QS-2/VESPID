import logging
from pathlib import Path, PurePosixPath
from os.path import dirname, abspath, expanduser, splitext, basename
from functools import cached_property
from argparse import ArgumentParser
from typing import List, AnyStr

from paramiko import SSHClient, SSHConfig
from paramiko.ssh_exception import AuthenticationException
from scp import SCPClient

from . import Neo4jConnectionHandler
from . import LOG_FORMAT, DB_IP, BOLT_PORT, EXPORT_FORMAT_CHOICES, IMPORT_DIR

HOME_DIR = expanduser("~")
SSH_CONFIG_DEFAULT = Path(HOME_DIR, ".ssh/config")

logging.basicConfig(
    format=LOG_FORMAT,
    level=logging.INFO,
    datefmt='%m/%d/%Y %H:%M:%S')

logger = logging.getLogger(__name__)


def export_scp_all(local_dir: str, output_file: str, ssh_remote_or_alias: str,
                   ssh_config_file: str = SSH_CONFIG_DEFAULT, neo4j_import_dir: str = IMPORT_DIR,
                   db_ip: str = DB_IP, bolt_port: str = BOLT_PORT, export_format: str = 'graphml'):
    """
    All-in-one wrapper for exporting everything and pulling via SCP once.

    Use the context of a Neo4jExportConnectionHandler object for ongoing connections!

    :param local_dir:
    :param output_file: filename to save locally and remotely.
                        Be mindful that it will overwrite remote files in the neo4j import dir
    :param ssh_remote_or_alias:
    :param ssh_config_file:
    :param neo4j_import_dir:
    :param db_ip:
    :param bolt_port:
    :param export_format:
    """
    _verify_filename_equals_basename(output_file)  # easier to not worry about recursive copying etc.
    h = Neo4jExportConnectionHandler(db_ip=db_ip, bolt_port=bolt_port,
                                     ssh_hostname_or_ssh_alias=ssh_remote_or_alias,
                                     ssh_config_file_loc=ssh_config_file)
    with h:
        logger.info(f"exporting all to {output_file} in format {export_format}... ")
        h.apoc_export_all(export_filename=output_file, export_format=export_format)
        remote_path = PurePosixPath(neo4j_import_dir, output_file).as_posix()  # assemble path for remote Linux instance
        logger.info(f"scp pulling to {local_dir} : {remote_path}")
        h.scp_get(remote_path, local_path=local_dir)


def export_scp_queries(query_iterable: List[AnyStr], local_dir: str, output_file: str, ssh_remote_or_alias: str,
                       ssh_config_file: str = SSH_CONFIG_DEFAULT, neo4j_import_dir: str = IMPORT_DIR,
                       db_ip: str = DB_IP, bolt_port: str = BOLT_PORT, export_format: str = 'graphml'):
    """
    All-in-one wrapper for exporting queries and pulling via SCP once.

    Use the context of a Neo4jExportConnectionHandler object for ongoing connections!

    :param query_iterable:
    :param local_dir:
    :param output_file: filename to save locally and remotely. Appends _1, _2, _3 etc. for each query after the first.
                        Be mindful that it will overwrite remote files in the neo4j import dir
    :param ssh_remote_or_alias:
    :param ssh_config_file:
    :param neo4j_import_dir:
    :param db_ip:
    :param bolt_port:
    :param export_format:
    """
    files_to_scp = []  # track name(s) of file(s) to pull
    base_filename, extension = splitext(output_file)  # split out and track for potentially creating multiple filenames
    _verify_filename_equals_basename(output_file)  # easier to not worry about recursive copying etc.
    handler = Neo4jExportConnectionHandler(db_ip=db_ip, bolt_port=bolt_port,
                                           ssh_hostname_or_ssh_alias=ssh_remote_or_alias,
                                           ssh_config_file_loc=ssh_config_file)
    with handler:
        for idx, line in enumerate(query_iterable):
            current_filename = f"{base_filename}_{idx}{extension}" if idx > 0 else output_file
            logger.info(f"on query {idx}: {line}")
            logger.info(f"exporting query {idx} to {current_filename} in format {format}... ")
            handler.apoc_export_query(query=line, export_format=export_format, export_filename=current_filename)
            files_to_scp.append(current_filename)
        remote_paths = [PurePosixPath(neo4j_import_dir, f).as_posix() for f in files_to_scp]
        logger.info(f"scp pulling to {local_dir} these remote paths: {remote_paths}")
        handler.scp_get(remote_paths, local_path=local_dir)


def _verify_filename_equals_basename(output_file):
    """For ease of managing files remotely and locally across SCP, check if an output file equals its basename."""
    if basename(output_file) != output_file:
        raise ValueError(f"specify only a basename for -o/--output-file "
                         f"(i.e., 'output.graphml' not 'path/to/output.graphml') "
                         f"This is for simplicity's sake managing files remotely and locally.")


class Neo4jExportConnectionHandler(Neo4jConnectionHandler):
    """
    Object used with 'with' context for maintaining ongoing neo4j and SSH connections for calling procedures and scp.

    Assumes a Linux remote instance and that SSH user has file permissions to Neo4j import directory.

    Assumes local computer running SSH agent, SSH config file and known_hosts set up.

    Example usage:

    handler = Neo4jExportConnectionHandler(...)
    with handler:
        handler.apoc_export_query(...)
        handler.scp_get(...)
    """
    def __init__(self, ssh_hostname_or_ssh_alias, db_ip=DB_IP, bolt_port=BOLT_PORT,
                 ssh_config_file_loc=SSH_CONFIG_DEFAULT, db_password=None):
        super().__init__(db_ip=db_ip, bolt_port=bolt_port, db_password=db_password)
        self.has_active_ssh_connection = False
        self.hostname_or_ssh_alias = ssh_hostname_or_ssh_alias
        self.ssh_config_file_loc = ssh_config_file_loc
        logger.debug(f"initialized {self.__class__.__name__}: "
                     f"db {self.db_ip}:{self.bolt_port} on {self.hostname_or_ssh_alias}")

    @cached_property
    def ssh(self):
        logger.debug(f"initializing ssh connection: {self.hostname_or_ssh_alias}")
        ssh = SSHClient()
        config = SSHConfig.from_path(self.ssh_config_file_loc)
        ssh.load_system_host_keys()
        try:
            conf = config.lookup(self.hostname_or_ssh_alias)
            logger.debug(f"found config {conf}")
            ssh.connect(conf['hostname'], username=conf["user"], key_filename=conf['identityfile'])
        except AuthenticationException as e:
            raise IOError(f"unable to connect to {self.hostname_or_ssh_alias}") from e
        self.has_active_ssh_connection = True
        return ssh

    @cached_property
    def scp(self):
        logger.debug(f"initializing scp connection: {self.hostname_or_ssh_alias}")
        return SCPClient(self.ssh.get_transport())

    def __enter__(self):
        pass  # half of contextlib interface; see __exit__

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.debug(f"{self.__class__.__name__} exiting context, exception info: {exc_type} {exc_val} {exc_tb}")
        if self.has_active_ssh_connection:
            logger.debug("closing SSH connections")
            self.scp.close()
            self.ssh.close()

    def scp_get(self, path_or_iterable, local_path=''):
        """
        Wrapper to calling get on cached SCP object
        :param path_or_iterable: single string file or multiple files
        :param local_path: where save the files locally. defaults to '', e.g., current working dir
        """
        logger.debug(f"scp_get: {path_or_iterable} to {local_path}")
        self.scp.get(path_or_iterable, local_path=local_path)

    def scp_put(self, path_or_iterable, path_two):
        """
        Wrapper to calling put on cached SCP object
        :param path_or_iterable: single string file or multiple files
        :param path_two: where save the files remotely
        """
        logger.debug(f"scp_put: {path_or_iterable} to {path_two}")
        self.scp.put(path_or_iterable, path_two)


if __name__ == "__main__":
    p = ArgumentParser(description="Issue an export command for all or querying the data, "
                                   "and SCP the result onto this local machine. "
                                   "Example usage: python neo4j_export_scp.py query "
                                   "-q \"MATCH (a:Publication) RETURN a LIMIT 5\" "
                                   "-o test_pub.graphml -r vespid_neo4j -vv")
    # main arguments in a parent parser to share among subparsers
    # thanks SO! https://stackoverflow.com/a/56595689
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument("-v", "--verbose", action='count', default=0,
                               help="count action for log level ERROR, WARNING, INFO, DEBUG")
    parent_parser.add_argument("-f", "--format", choices=EXPORT_FORMAT_CHOICES, default='graphml')
    parent_parser.add_argument("--db-ip", default=DB_IP,
                               help=f"if given, use this as location of remote db server. defaults to {DB_IP}")
    parent_parser.add_argument("--bolt-port", default=BOLT_PORT,
                               help=f"if given, use this as location of remote db server. defaults to {BOLT_PORT}")
    parent_parser.add_argument("-o", "--output-file", required=True,
                               help="basename of file "
                                    "(choose a unique name, as it "
                                    "will overwrite remote files in the neo4j import dir)")
    parent_parser.add_argument("-d", "--output-dir", help="where save the file locally",
                               default=dirname(abspath(__file__)))
    parent_parser.add_argument("-r", "--ssh-remote-or-alias", required=True,
                               help="url, ip, or SSH remote where neo4j lives; calls scp")
    parent_parser.add_argument("--ssh-config-file", default=SSH_CONFIG_DEFAULT,
                               help=f"location of SSH config file. defaults to {SSH_CONFIG_DEFAULT}")
    parent_parser.add_argument("--debug-paramiko", action="store_true",
                               help="default to logging.INFO for paramiko; pass this to log paramiko at logging.DEBUG")
    # subparsers, one per mode 'all' and 'query
    subparsers = p.add_subparsers(dest='mode', help="mode for exporting")
    subparsers.required = True
    _query = subparsers.add_parser("query", parents=[parent_parser])
    group = _query.add_mutually_exclusive_group(required=True)
    group.add_argument("-q", "--queries", nargs="+", help="string list of queries; "
                                                          "composes output-file with _0, _1, _2, etc.")
    group.add_argument("-Q", "--query-file", help="filename to load queries, one query per line; "
                                                  "composes output-file with _0, _1, _2, etc.")
    _all = subparsers.add_parser("all", parents=[parent_parser])
    args = p.parse_args()
    log_level = logging.ERROR - (args.verbose * 10)
    logger.setLevel(log_level)
    # paramiko doesn't respect this setting. dumb. but it also prints way too much debug, so...
    if logger.getEffectiveLevel() <= logging.INFO:
        paramiko_level = logging.DEBUG if args.debug_paramiko else logging.INFO
        logging.getLogger('paramiko').setLevel(max(log_level, paramiko_level))
    else:
        logging.getLogger('paramiko').setLevel(max(log_level, logging.WARNING))
    logger.info(f"args: {args}")

    if args.mode == 'all':
        export_scp_all(db_ip=args.db_ip, bolt_port=args.bolt_port,
                       ssh_remote_or_alias=args.ssh_remote_or_alias,
                       ssh_config_file=args.ssh_config_file,
                       local_dir=args.output_dir, output_file=args.output_file)
    elif args.mode == 'query':
        if args.query_file:
            queries_iter = open(args.query_file)
        elif args.queries:
            queries_iter = args.queries
        else:
            raise ValueError(f"this shouldn't happen w/mutually exclusive but required, but make the warning happy")
        export_scp_queries(query_iterable=queries_iter,
                           db_ip=args.db_ip, bolt_port=args.bolt_port,
                           ssh_remote_or_alias=args.ssh_remote_or_alias,
                           ssh_config_file=args.ssh_config_file,
                           local_dir=args.output_dir, output_file=args.output_file)
    else:
        raise ValueError(f"mode not implemented: {args.mode}")
