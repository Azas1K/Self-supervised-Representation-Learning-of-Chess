from time import sleep
import re
import requests, zipfile, io
from bs4 import BeautifulSoup

urls = 'https://www.pgnmentor.com/files.html'
grab = requests.get(urls)
soup = BeautifulSoup(grab.text, 'html.parser')

links = []
# opening a file in write mode
f = open("test1.txt", "w")
# traverse paragraphs from soup
for link in soup.find_all("a"):
    data = link.get('href')
    if data:
        links.append(data)
        f.write(data)
        f.write("\n")

f.close()


myfiles=[]
for l in set(links): #you can also iterate through br.forms() to print forms on the page!
    if re.search(".*players/.*\.zip$" ,str(l)): #check if this link has the file extension we want (you may choose to use reg expressions or something)
        myfiles.append(l)

myfiles = sorted(myfiles)
f = open("test2.txt", "w")
for i in myfiles:
    f.write(i)
    f.write("\n")
f.close()


def download_file(url):
    path = "https://www.pgnmentor.com/"+url
    r = requests.get(path, allow_redirects=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("database/")


for url in myfiles:
    sleep(1) #throttle so you dont hammer the site
    download_file(url)
