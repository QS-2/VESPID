import numpy as np

def preprocess_text(documents, language='english'):
    """ 
    Basic preprocessing of text, based on that which is done by BERTopic package.
    
    Steps:
        * Lower text
        * Replace \n and \t with whitespace
        * Only keep alpha-numerical characters
        
        
    Parameters
    ----------
    documents: pandas Series of strings, one per document.
    
    language: str. Only supported value right now is 'english'. Anything
        else just causes some string processing to be skipped.
        
        
    Returns
    -------
    pandas Series of strings processed accordingly.
    """
    cleaned_documents = documents.fillna("").str.lower()
    #TODO: make this more efficient with a single regex
    cleaned_documents = cleaned_documents.str.replace("\n", " ").str.replace("\t", " ")
    
    if language == "english":
        cleaned_documents = cleaned_documents.str.replace(r'[^A-Za-z0-9 ]+', '', regex=True)
    
    return cleaned_documents