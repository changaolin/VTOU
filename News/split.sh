#!/usr/bin/env bash
echo "news_sohusite_xml.dat 文件在百度云盘，地址见README"
inp="news_sohusite_xml.dat"
cat $inp | iconv -f gbk -t utf-8 -c | grep "<content>"  > content.txt
cat $inp | iconv -f gbk -t utf-8 -c | grep "<url>"  > url.txt
cat $inp | iconv -f gbk -t utf-8 -c | grep "<contenttitle>"  > title.txt
