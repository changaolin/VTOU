#!/usr/bin/env bash
inp="news_sohusite_xml.dat"
cat $inp | iconv -f gbk -t utf-8 -c | grep "<content>"  > content.txt
cat $inp | iconv -f gbk -t utf-8 -c | grep "<url>"  > url.txt
cat $inp | iconv -f gbk -t utf-8 -c | grep "<contenttitle>"  > title.txt
