import urllib.request
import ssl
from urllib.parse import quote
import string
appcode = '***'
querys = '**'
myphone='**'
send = False
def sendtoPhone(msg='训练结果',appcode=appcode,querys=querys):
    if send == False:
        return
    host = 'https://feginesms.market.alicloudapi.com'
    path = '/codeNotice'
    method = 'GET'
    querys = querys.replace('msg', msg)
    querys = querys.replace('myphone', myphone)
    print(querys)

    bodys = {}
    url = host + path + '?' + querys
    newurl = quote(url, safe=string.printable)
    request = urllib.request.Request(newurl)
    request.add_header('Authorization', 'APPCODE ' + appcode)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    response = urllib.request.urlopen(request, context=ctx)
    content = response.read()
    if (content):
        print(content.decode('UTF-8'))
