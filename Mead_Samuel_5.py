# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:01:50 2020

@author: Samuel
"""
def setup():
    import os
    import json
    nltk.download() # if already done just comment out or exit the downloader when it pops up
    os.mkdir('mead_py_5') 
    os.chdir('mead_py_5') 
    results = open('results.json', 'w+')
    entities = open('entities.json','w+')
    #Please read the requirements.txt file to ensure Newspaper3k and BS4 are installed

        
def bbc_scraper(url):
    import nltk 
    from newspaper import Article
    from bs4 import BeautifulSoup
    import json
    article = Article(url)
    article.download()
    article.parse()
    article.nlp()
    """The part below threw me off - eventually I ended up taping a Bs4 parse 
    over in case the url is a BBC article so it remains flexible with the rest 
    of the interet. Using Article it returned "None" for he publishdate. Going 
    through the back end Article code, the tags which it looks for in the 
    html do include "datePublished" but figured the issue was BBC using
    "datePublished":"2020....." rather than "datePublished=........" If anyone
    has a solution please let me know  as this made things messy. This may need
    to be periodically edited if BBC change their html formatting as i ended
    slicing the index locations of where this is in the code, but this seems to
    work robustly enough.
    """
    if 'bbc' in url: 
        soup = BeautifulSoup(article.html, 'lxml')
        a = [article.html]
        att=str(a)
        b= att.find('datePublished')
        c = b
        while att[c] != ',':
            c=c+1
        publish_date = (att[b+16:b+26]) #Originally I iterated to make it dynamic, but this way made the string neater. 
    else:
        publish_date = str(article.publish_date)
        
    data = {'url':article.url,
            'title': article.title,
            'Publish_date': publish_date,
            'Text': article.text}
    results_json = json.dumps(data)           
    return results_json



def extract_entities(string):
    import nltk 
    import json
    tags = nltk.pos_tag(string.split()) # get the tags
    disp = nltk.ne_chunk(tags) # I continued to call it disp because of the draw() addition being how i got around it
    People = ['People:'] #set some arrays to split each entity type if required
    Organization = ['Organization']
    Places=['Places']
    
    for ne in disp.subtrees():
        if ne.label()=='PERSON':
            for item in ne:
                People.append(item[0]) #just to trim and slightly neaten what gets returned - I didn't think you wanted to see each tag
        if ne.label()=='ORGANIZATION':
            for item  in ne:
                Organization.append(item[0])                
        if ne.label()=='LOCATION':
            for item in ne:
                Places.append(item[0])
        if ne.label()=='FACILITY':
            for item in ne:
                Places.append(item[0])
        if ne.label()=='GPE':
            for item in ne:
                Places.append(item[0])
    
    data = [People, Organization, Places]
    entities_json = json.dumps(data)
    with open('entities.json','w') as f:
        json.dump(entities_json, f)
    return(entities_json)
    
    
def test_scraper_function():
    """Test logic: has the scraper_function been able to iterate over a few URLs and provide un-muddled data?
    The array a takes information required for each link and appends, then writes to the JSON file. 
    If it can find the same amount of a certain attribute, i.e. date published, it has managed to 
    iterate and store the information. Sanity check: print the test data to make sure it's not 
    written the same thing n times. Publish_date was chosen as 'url', 'Title' and 'Text' were common in article bodies."""
    import json
    url_list = ['https://www.bbc.co.uk/news/extra/g7fg26ab8b/drawings-from-lockdown',
                'https://www.bbc.co.uk/news/uk-52405852','https://www.bbc.co.uk/news/world-us-canada-52384622',
                'https://www.channel4.com/news/how-to-get-out-of-lockdown-scotland-sets-out-its-plan']
    a = []
    for url in url_list:
        b= bbc_scraper(url)
        a.append(b)
    with open('results.json','w') as json_file:
            json.dump(a, json_file)  
    with open('results.json', 'r') as result_trial:
        test_data = json.load(result_trial)
        x = str(test_data)
        y = x.count('Publish_date')
        z = len(url_list)
        if y==z:
            print("Seems to work")
        else:
            print("Have another look")
        #print(y,z)
        #print(test_data)
        
def test_entities_function():
    """"Test logic: if given the string will the NER provide the correct output from the JSON file?"""
    string = """What did President Trump say?
    During Thursday's White House coronavirus task force briefing, an official presented the results of US government research that indicated coronavirus appeared to weaken faster when exposed to sunlight and heat.

The study also showed bleach could kill the virus in saliva or respiratory fluids within five minutes, and isopropyl alcohol could kill it even more quickly.

Mr Trump then hypothesised about the possibility of using a "tremendous ultraviolet" or "just very powerful light" on or even inside the body as a potential treatment """
    extract_entities(string)
    with open('entities.json', 'r') as entities_trial:
        test_data = json.load(entities_trial)
    print(test_data)
    
