# Adapted from: https://github.com/ruhomor/Meme-Generator/blob/master/data_scraper.py

from bs4 import BeautifulSoup as bs
from bs4 import SoupStrainer as ss
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import requests
import argparse
import pandas as pd

base_url = 'https://memegenerator.net'
meme_template_url = f'{base_url}/memes/popular/alltime/'

def get_bs(url, parse_only=None):
  r = requests.get(url)
  return bs(r.text, 'html.parser', parse_only=parse_only)

def save_meme_template(save_dir, url):
  r = requests.get(url, stream=True)
  name = url.split('/')[-1]
  with open(f'{save_dir}/{name}', 'wb') as outfile:
    outfile.write(r.content)

def process_meme_template_page(meme_template, j):
  dfs = []
  meme_href = meme_template.find('a')['href']
  page_url = f"{base_url}/{meme_href}" if j == 0 else f"{base_url}/{meme_href}/images/popular/alltime/page/{str(j + 1)}"
  memes = get_bs(page_url, parse_only=ss('a', href=re.compile('instance')))
  for meme in memes:
    s = get_bs(f"{base_url}/{meme.get('href')}", parse_only=ss(class_='w100p'))
    caption = s.find('img')['alt']
    dfs.append(pd.DataFrame([[meme_href[1:], caption.split('-')[1][1:].lower()]], columns=['type', 'caption']))
  return pd.concat(dfs, ignore_index=True)

def main(n_pages_meme_types, n_pages_meme_egs, save_dir, outfile_name):
  dfs = []
  for i in range(n_pages_meme_types):
    url = meme_template_url if i == 0 else f'{meme_template_url}/page/{str(i + 1)}'
    meme_templates = get_bs(url, parse_only=ss(class_='char-img'))
    for j, meme_template in enumerate(meme_templates):
      print(f'{i}-{j}')
      save_meme_template(save_dir, meme_template.find('img')['src'])
      with ThreadPoolExecutor() as executor:
        futures = [executor.submit(lambda i: process_meme_template_page(meme_template, i), i) for i in range(n_pages_meme_egs)]
        for future in as_completed(futures):
          dfs.append(future.result())
  df = pd.concat(dfs, ignore_index=True)
  df.to_csv(f'{outfile_name}.csv')

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Helper script to download meme templates and captions from memegenerator.net')
  parser.add_argument('n_pages_meme_types', type=int, help='number of meme template pages to scrape')
  parser.add_argument('n_pages_meme_egs', type=int, help='number of memes per template to scrape')
  parser.add_argument('--save_dir', type=str, default='meme_templates', help='local directory to save scraped meme templates')
  parser.add_argument('--outfile_name', type=str, default='captions', help='CSV filename for scraped captions')
  args = parser.parse_args()
  main(args.n_pages_meme_types, args.n_pages_meme_egs, args.save_dir, args.outfile_name)
