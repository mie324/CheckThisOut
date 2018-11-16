import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy
import spacy
from lxml import html

spacy_en = spacy.load('en')

def scrap_abstract(game_name):# the input to this function should be of the form 'Sid+Meier+Civilization+V'
    page_url='https://store.steampowered.com/search/?term='

    #game_name='Sid+Meier+Civilization+V'
    
    COOKIES = {'birthtime': '283993201','mature_content': '1',}
    page=requests.get(page_url+game_name,cookies=COOKIES)
    #page=requests.get(page_url+game_name)

    soup=BeautifulSoup(page.content,'html.parser')

    all_links=soup.find_all('a')
    
    t_link='empty'
    
    
    for link in all_links:
        if 'https://store.steampowered.com/app/'in link.get('href') and '/353370/' not in link.get('href') and '/353380/' not in link.get('href') and '/358040/' not in link.get('href'):
            t_link=link.get('href')
            #print(t_link)
            break
    if t_link=='empty':
        print('url not working')
        print(page_url+game_name)
        print('\n')
        return ''
            
    # at this point, we have the link to the game intro page.
    
    page=requests.get(t_link)
    soup=BeautifulSoup(page.content,'html.parser')
    
    start_header='empty1'
    end_header='empty2'
    all_header=soup.find_all('h2')
    
    found_start=0
    for header in all_header:
        if found_start==1:
            end_header=header
            break
        if found_start==0:
            if header.get_text()=='About This Game':
                start_header=header
                found_start=1

    if start_header=='empty1':
        print('there is nothing')
        print(page_url+game_name)
        print('\n')
        return ''
    
    #print(start_header.parent.get_text())
    abstract=start_header.parent.get_text()
    abstract=abstract.replace('About This Game','')
    abstract=" ".join(abstract.split())
    #print(abstract)
    return abstract


    
def scrap_all_abstract(start,end):
    file=pd.read_csv('mostplayedgames.csv')
    file=file.values.tolist()
    #for i in range(len(file)):
        #print(file[i][1])
    ans_list=[]
    #for i in range(len(file)):
    for i in range(start ,end+1):
        name=file[i][1]
        print(i,name)
        temp_name=tokenizer(name)
        temp_name2=temp_name[0]
        for j in range(len(temp_name)-1):
            temp_name2+='+'+temp_name[j+1]
        abstract=scrap_abstract(temp_name2)
        ans=[name,abstract]
        ans_list.append(ans)
    ans_list=pd.DataFrame(ans_list)
    ans_list.to_csv(str(start)+'_'+str(end)+'.csv')
    
    
    
def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def scrap_age_check():
    session_requests = requests.session()
    login_url='https://store.steampowered.com/login/?redir=agecheck%2Fapp%2F202970%2F%3Fsnr%3D1_7_7_151_150_1&redir_ssl=1'
    #result = session_requests.get(login_url)
    payload={'username':'chengchengcheng16','password':''}
    url='https://store.steampowered.com/app/202970/Call_of_Duty_Black_Ops_II/'
    page = session_requests.post(login_url, data = payload, headers = dict(referer=login_url))
    page=session_requests.get(url,headers = dict(referer = url))
    #page=requests.get(page_url+game_name)
    

    soup=BeautifulSoup(page.content,'html.parser')

    all_links=soup.find_all('a')
    
    t_link='empty'
    
    
    for link in all_links:
        if 'https://store.steampowered.com/app/'in link.get('href') and '/353370/' not in link.get('href') and '/353380/' not in link.get('href') and '/358040/' not in link.get('href'):
            t_link=link.get('href')
            #print(t_link)
            break
    print(t_link)
    
    
    
    

if __name__=='__main__':
    #ans=scrap_abstract('Sid+Meier+Civilization+V')
    #print(ans)
    scrap_all_abstract(1,500)
    #scrap_age_check()
    
    #test_list=[[1,2],[3,4],[5,6],[7,8]]
    #test_list=pd.DataFrame(test_list)
    #test_list.to_csv('test_save.csv')

