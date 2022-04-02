# VESPID

A project combining network analysis with text-based clustering with scientific literature to identify waxing and waning scientific disciplines and other interesting questions for research.

For more details on how and why the contents of this repo were built, please see [our paper!](https://arxiv.org/abs/2202.12913)

## Getting Started

Should be as easy as `pip install -r requirements.txt` and then, possibly, you'll also need to do `pip install .` from the repo root to install the `vespid` package.

Note that a lot of this work was done in a Linux-based Docker container environment, so if any details look particularly Linux-y to you, that's why.

This project directory structure is a variant on that provided by [cookiecutter data science](https://drivendata.github.io/cookiecutter-data-science/).

## Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    |
    ├── vespid             <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │                     predictions
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations

## Python Code Installation

If actively developing code for the package, we recommend installing it via `pip install -e .` when in the project root directory. This will make it be in editable mode so any changes should be auto-installed on your machine. Make sure any notebook you may be using also has 

```
import autoreload
%load_ext autoreload
%autoreload 2
```

in its imports, so the edits are auto-loaded into the kernel too.

## The Technology Stack

A huge portion of this code assumes you have an instance of Neo4j (we used Community Edition) running on a server that you can access. We used Neo4j for the network-based analyses presented herein as well as a single source of "data truth" for all collaborators and developers. 

## The Data

Unfortunately, our dataset was proprietary as it utilized the Web of Science, but much of what we did here could be replicated via an open dataset like Semantic Scholar (which we also used). 

## AWS Serverless Infrastructure

We used AWS Batch to run GPU- and CPU-based loads for much of this project, in particular the data engineering bits converting raw bibliometric data into a Neo4j graph format and training and optimizing UMAP+HDBSCAN-based clustering pipelines. We recommend you do the same.

## Neo4j Infrastructure

In case you're interested in setting up a Neo4j server on AWS like we did, here's some info for you.

### Amazon EC2 Instance

#### [Installation/Setup](https://neo4j.com/docs/operations-manual/current/installation/linux/rpm/#linux-rpm)

1. Create an instance (e.g. t2.large) with at least 2 GB of memory, ideally 16 GB, using the Amazon Linux 2 AMI
2. `ssh` in
3. `sudo rpm --import https://debian.neo4j.com/neotechnology.gpg.key`
4. Open up a text editor like `nano` and create the file `/etc/yum.repos.d/neo4j.repo` with the contents:
    ```
    [neo4j]
    name=Neo4j RPM Repository
    baseurl=https://yum.neo4j.com/stable
    enabled=1
    gpgcheck=1
    ```
5. `sudo amazon-linux-extras enable java-openjdk11` to ensure you have Java 11 enabled
6. `sudo yum install neo4j` for the latest version or `sudo yum install neo4j-<version_num>` for a specific version
7. `nano /etc/neo4j/neo4j.conf`
    * Uncomment `dbms.connectors.default_listen_address=0.0.0.0` so it will accept IPs from outside of localhost
    * Make sure `dbms.connector.bolt.enabled=true` (should be by default)
    * Consider setting `dbms.connector.https.enabled=true`, if you have an SSL policy/certificate you can also provide, and then set the corresponding HTTP setting to false (so you only send authentication info over the secure wire using the 7473, instead of unsecured 7474, port)
8. Go to `public_IP:7474` in your browser and use username=neo4j, password=neo4j for the first-time login (you'll be prompted to create a new one going forward)
9. Follow Neo4j's helpful guides on ingesting data into your new database!

### Backing Up the Database

Often useful for when you want to switch out datasets (if using Community Edition, wherein only a single database is allowed, or if you are storage-constrained). 

1. (as neo4j user) `neo4j stop`
2. `neo4j-admin dump --database=neo4j --to=/dumps/neo4j/neo4j-<db_nickname>-<datetimestamp>.dump`
    * If the directory `/dumps/neo4j/` doesn't exist yet, switch to ec2-user and run `sudo mkdir /dumps/ && sudo mkdir /dumps/neo4j/ && sudo chown neo4j:neo4j /dumps/neo4j/`
    * Note that 150M nodes and 523M edges results in a compressed dump file of around 61 GB
3. Jump over to ec2-user
4. Copy your dump file somewhere helpful, like an S3 bucket
5. `sudo su - neo4j`
6. Import/ingest data as needed
7. `neo4j start && tail -f /var/log/neo4j/neo4j.log`

## Upgrading the DBMS

Note that these steps have some explicit version numbers for software/packages used. Make sure you update to the latest versions of these before running any of this code!

1. Follow steps 1-4 in the instructions above for backing up the database
2. Copy the old 
    1. `sudo cp /etc/neo4j/neo4j.conf /etc/neo4j/neo4j_<current_date_in_form_MM-DD-YYYY>.conf`. 
    * This ensures we can throw any custom config settings into the upgraded DB config, in case it doesn't do so automatically during the upgrade
3. `sudo yum update` to update the core DBMS as ec2-user (and anything else needing it!)
4. `sudo su - neo4j` to switch to neo4j user
5. `nano /etc/neo4j/neo4j.conf` and set `dbms.allow_upgrade=true` as well as `dbms.mode=SINGLE` (you'll have to add the second one as a new line likely)
        * Full form of dbms.mode entry should be: 
        ```
        # Makes sure the system database is upgraded too
        dbms.mode=SINGLE
        ```
6. `cd /var/lib/neo4j/plugins/` and install updated versions of the plugins that will work with the Neo4j version you're installing
    1. `rm <apoc_jar_file> && rm <gds_jar_file>` to remove old plugins so system isn't confused between old and new versions at startup
    2. `wget https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/4.4.0.1/apoc-4.4.0.1-all.jar`
    3. `wget https://s3-eu-west-1.amazonaws.com/com.neo4j.graphalgorithms.dist/graph-data-science/neo4j-graph-data-science-1.8.1-standalone.zip && unzip neo4j-graph-data-science-1.8.1-standalone.zip`
    4. `rm neo4j-graph-data-science-1.8.1-standalone.zip`
7. `cd ..` to get back to `/var/lib/neo4j/` and then `neo4j start && tail -f /var/log/neo4j/neo4j.log`: the process of starting the database *should* cause it to actually perform the upgrade, which you'll see in the logs when it spins up
8. Kick the tires on the newly spun-up DBMS:
    1. `CALL dbms.components()` to check that you're at the version of the core DB you're expecting
    2. `RETURN gds.version()`
    3. `RETURN apoc.version()`
    4. `MATCH (n) RETURN COUNT(n)` just to be sure!
9. `nano /etc/neo4j/neo4j.conf` and set `dbms.allow_upgrade=false` and comment out `dbms.mode=SINGLE`
10. `neo4j restart && tail -f /var/log/neo4j/neo4j.log`

### Exporting and Importing (and Quirks)

You can call export and import procedures from `apoc` (see below), moving data from remote instance to your local and back, etc. We've created `vespid.data.neo4j.export_scp` script to help you with this (exporting and scp'ing to your local). Note that you'll need to add a line at the end of `/etc/neo4j/neo4j.conf` that says `apoc.export.file.enabled=true` in order to make this work.

To import the data into a new database via a file on disk, make sure `apoc.import.file.enabled=true` in `neo4j.conf`. A simple approach to importing at that point (after moving the relevant graphML file into the neo4j `import/` directory) is to run `CALL apoc.import.graphml("file:///<filename.graphml>", {readLabels: 'true'})`.

Note that you can also load data programmatically in python, if that makes more sense, via `Neo4jConnectionHandler.insert_data()`.

### Deleting Database for a Fresh Start

The assumption in this section is that we're trying to overwrite an old version of an existing database (e.g. the default `neo4j` database), likely because we're using Community Edition and can't access more than one database at a time.

1. (as neo4j user in EC2 instance in /var/lib/neo4j/) `neo4j stop`
    * Note that `neo4j status` will report that Neo4j isn't running if you run that command without being the neo4j user first.
2. `rm -rf data/databases/neo4j/ && rm -rf data/transactions/neo4j/`
    * This deletes the default database contents so we can import into a fresh database instance
2. `neo4j-admin import <details here>` *or* `neo4j-admin load <details here>` if loading from a Neo4j database dump file
3. `neo4j start && tail -f /var/log/neo4j/neo4j.log`

### Installing Plugins

#### [Graph Data Science Library](https://neo4j.com/docs/graph-data-science/current/installation/)

1. `cd /var/lib/plugins/` if you're not already there
2. `wget https://s3-eu-west-1.amazonaws.com/com.neo4j.graphalgorithms.dist/graph-data-science/neo4j-graph-data-science-1.7.2-standalone.zip`
    * This is the latest version as of 2/15/2021, but check [the Neo4j Download Center](https://neo4j.com/download-center/) for the latest version before downloading - can get URL by right-clicking and copying URL from within browser.
2. `unzip <file_you_just_got>`
3. `sudo mv <unzipped_jar_file> /var/lib/neo4j/plugins/`
4. `nano /etc/neo4j/neo4j.conf`
5. Uncomment and modify relevant line to be `dbms.security.procedures.unrestricted=gds.*`
    * May also have other procedure groups listed here for other plugins, in a comma-separated fashion. If so, leave those as they are and just add the `gds.*` bit.
6. `neo4j restart`
7. In Cypher: `RETURN gds.version()` to verify you got what you were looking for
8. In Cypher: `CALL gds.list()` to see all procedures/algorithms available to you

#### APOC [here](https://neo4j.com/graphacademy/training-basic-admin-40/09-configuring-plugins/#_example_the_apoc_plugin) and [here](https://github.com/neo4j-contrib/neo4j-apoc-procedures).

1. `cd /var/lib/plugins/` if you're not already there
2. `wget https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/4.3.0.4/apoc-4.3.0.4-all.jar`
3. `mv <file_you_just_got> plugins/`
4. `neo4j restart` && tail -f /var/log/neo4j/neo4j.log
5. `RETURN apoc.version()` to check that it's installed as expected

#### [Neo4j-Arrow](https://github.com/neo4j-field/neo4j-arrow)
1. Run the below uncommented commands in your current shell, and for later re-runs, create a `.profile` file owned by the `neo4j` user inside of `$NEO4J_HOME` on the server you're installing, with the following contents (ensuring you've installed certificates to these locations, setting up SSL)
```
# neo4j arrow config, per https://github.com/neo4j-field/neo4j-arrow#configuration-%EF%B8%8F
export HOST="0.0.0.0"
export ARROW_TLS_PRIVATE_KEY="$NEO4J_HOME/certificates/private.key"
export ARROW_TLS_CERTIFICATE="$NEO4J_HOME/certificates/public.crt"
# # these are options that might help if you're running into issues:
# first remove what you might have set already for a clean slate...
unset MAX_MEM_GLOBAL MAX_MEM_STREAM ARROW_MAX_PARTITIONS ARROW_BATCH_SIZE
# # now set options!
# export MAX_MEM_GLOBAL=24
# export MAX_MEM_STREAM=8
# export ARROW_MAX_PARTITIONS=6
# export ARROW_BATCH_SIZE=500
``` 
2. Install the jar plugin from [the latest release page](https://github.com/neo4j-field/neo4j-arrow/releases/latest)
   1. Download the jar
   2. As the `neo4j` user, move the file into `$NEO4J_HOME/plugins/` and change ownership to match other files (e.g., `chown neo4j:neo4j`)
3. Restart the server with the new configuration
   1. If you didn't before, run the commands in your new `.profile` with `source .profile`
      1. The `.profile` above does this for you, but if you're manually removing variables, don't forget you have to `unset`, not just comment the `export` line in `.profile`!
   2. `neo4j restart` && tail -f /var/log/neo4j/neo4j.log
   3. Verify that the log contains something like `INFO org.neo4j.arrow.App - server listening @ grpc+tcp://0.0.0.0:9999`
   4. Test with a dummy query call to `Neo4jConnectionHandler(...).cypher_query_to_dataframe(..., drivers='neo4j-arrow')`!

### Using Your New Database

The most basic elements of using the database come from `vespid.data.neo4j_tools`. Here's some basic usage:

```
from vespid.data.neo4j_tools import Neo4jConnectionHandler
graph = Neo4jConnectionHandler(db_ip=db_ip, db_password=db_password)

query = """
MATCH (p:Publication)
WHERE p.title IS NOT NULL
RETURN p.title AS PublicationTitle LIMIT 5
"""

df = graph.cypher_query_to_dataframe(query)
```

This code will return the graph database query results as a pandas DataFrame. If you don't recognize the query language used above, plug "Neo4j Cypher" into your search engine of choice and you'll find info on it.

### Installing Python

Amazon Linux 2 AMI comes with python 2 by default, so we need to get some basic functionality going so we can do fun things like use `glances` to monitor system operation!

1. [As ec2-user] `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
2. `bash Miniconda3-latest-Linux-x86_64.sh`
3. Agree to all terms and allow the default install directory (which should be in ec2-user home or some such)
4. Allow it to initialize conda
5. `exit` to drop the SSH connection and then SSH back in (to refresh the shell with `conda` commands)
6. `pip install glances`

## MLFlow Model Tracking and Registry

This project involves training and using models in multiple places, with different and/or connected pipelines for training and serving. As such, for the purposes of maximal collaboration, we need to maintain metadata, experimental records, and artifacts from the models developed so we can use them more effectively. 

The [remote MLFlow tracking server](https://mlflow.org/docs/latest/tracking.html) is designed to track experiments such as entirely new training runs, hyperparameter tuning experiments, etc. It also serves to maintain an authoratative source of all trained models that we found to be optimal after tuning and evaluation. These models should be used in the proper context across different contexts. To this end, we setup our own tracking server backed by an S3 artifact store and recommend that you do the same!

### Installation
#### Tracking Server

Note that we've taken elements from [these](https://betterprogramming.pub/how-to-provision-a-cheap-postgresql-database-in-aws-ec2-9984ff3ddaea) [articles](https://towardsdatascience.com/setup-mlflow-in-production-d72aecde7fef) to set all of this up.

1. Spin up a tracking server with a database (postgresql ideally) you can use (e.g. AWS EC2 t2.medium with Amazon Linux 2)
    * Add IAM role that will grant access to S3 for artifact storage (e.g. `ecsInstanceRole`)
    * Make sure security group has port 5432 open for postgresql database
    * Tag with `{project: <org_name>, type: database, database_type: postgresql}`
    * Associate an Elastic IP with it that can then be added to our domain via a DNS type-A record
        * Note that the default port for MLFlow is 5000
2. SSH into your server
3. Setup the postgresql server
    1. `sudo apt-get update -y && sudo apt-get upgrade -y`
    2. `sudo amazon-linux-extras install postgresql13`
    3. `sudo yum install postgresql postgresql-server`
    4. `sudo /usr/bin/postgresql-setup --initdb`
    5. `sudo systemctl enable postgresql.service`
    6. `sudo systemctl start postgresql.service`
    7. `sudo -u postgres psql`
        "CREATE ROLE mlflow_user;"
        "CREATE DATABASE mlflow_db;"
        "CREATE USER mlflow_user WITH ENCRYPTED PASSWORD 'mlflow';"
        "GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;"
        "\q"
    8. `sudo nano /var/lib/pgsql/data/postgresql.conf`
        * Look for "listen_addresses" and replace the entry there with  `listen_addresses = '*'`
    9. `sudo nano /var/lib/pgsql/data/pg_hba.conf`
        * Add this line at the ***top*** of the host/ident/etc. config stuff so it's read first (one tab between entries, except two tabs between the final two entries):
            `host    all             all              0.0.0.0/0                       md5`
    10. `sudo systemctl restart postgresql`
4. `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh`
    * Agree to all terms and allow the default install directory (which should be in ec2-user home or some such)
    * Allow it to initialize conda
5. `exit` to drop the SSH connection and then SSH back in (to refresh the shell with `conda` commands)
6. `sudo yum install gcc`
6. `pip install glances mlflow psycopg2-binary boto3`
7. Setup the mlflow tracking server as a failure-protected service
    1. `sudo nano /etc/systemd/system/mlflow-tracking.service` and write the contents of `mlflow/mlflow-tracking.service` and save
    2. `sudo systemctl daemon-reload`
    3. `sudo systemctl enable mlflow-tracking`
    4. `sudo systemctl start mlflow-tracking && sudo systemctl status mlflow-tracking`

## Open Source Project Opportunities

Many, many open source libraries were used to build this project, and we thank every contributor involved in those for their contributions! Additionally, we developed a lot of new tools along the way in this project, but have not yet had the opportunity to build them out as standalone libraries of their own. So this section will serve as our "to do" list for those. Help in spinning these out into their own repos or cleaning them up to submit to existing relevant repos would be most welcome!

1. `vespid.models.optuna` contains new classes of `Criterion`, `Hyperparameter`, and `Objectives` that enable a much more flexible approach to `optuna`-based Bayesian hyperparameter optimization, and thus would likely be a fantastic addition to the `optuna` library.
2. `vespid.models.neo4j_tools.Neo4jConnectionHandler` includes a bunch of handy methods for efficiently exploring new graphs as well as multi-driver support that allows users to pick the driver that best suits their needs. Thus far, we've found:
    * The native/official Neo4j driver (published by Neo4j themselves) is great for data inserts
    * `py2neo` is great at read-only queries at small scale
    * `neo4j-arrow` is the fastest by far for read-only queries (and likely for inserts as well, but we haven't tested this). However, it can fail for very large (e.g. millions of records) queries. That said, it is an extremely new library and bound to improve in leaps and bounds in the near future.
3. `vespid.models.visualization` has some helpful code on visualizing very large graphs via edge-bundling techniques and zoom-context-aware dynamic resolutions, enabled largely by the [datashader](https://datashader.org/) library.
4. `vespid.models.mlflow_tools` has some nice helper functions that make quickly setting up new MLFlow experiments a breeze. These enhancements would like make a nice PR to the `mlflow` library, once they've been cleaned up a bit.
5. We created `vespid.pipeline` to make it quick and easy (and hopefully intuitive) to set up data pipelines. There may be some useful concepts in here for projects like Apache Airflow, but it's also possible we simply could have used that, had we had the time to learn it :P
6. Along the same lines as the earlier item about `vespid.models.optuna`, `vespid.models.batch_cluster_tuning.py` has some interesting ideas we explore more in the paper referenced at the top of the README. Specifically, we take a multi-object Bayesian hyperparameter optimization approach to finding robust and reproducible HDBSCAN-based clustering solutions. It's possible, with more experimentation on novel datasets, that this could become a clustering library unto itself or an addition to a larger clustering library.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
