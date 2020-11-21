# Adapted from: https://github.com/ruhomor/Meme-Generator/blob/master/data_scraper.py

from bs4 import BeautifulSoup as bs
from bs4 import SoupStrainer as ss
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import requests
import argparse
import pandas as pd

base_url = 'https://imgflip.com/'

def get_bs(payload, parse_only=None):
  r = requests.get(base_url, params=payload)
  return bs(r.text, 'html.parser', parse_only=parse_only)

def save_meme_template(save_dir, url):
  r = requests.get(url, stream=True)
  name = url.split('/')[-1]
  with open(f'{save_dir}/{name}', 'wb') as outfile:
    outfile.write(r.content)

def process_month_page(month, j):
  try:
    dfs = []
    meme_href = meme_template.find('a')['href']
    page_url = f"{base_url}/{meme_href}" if j == 0 else f"{base_url}/{meme_href}/images/popular/alltime/page/{str(j + 1)}"
    memes = get_bs(page_url, parse_only=ss('a', href=re.compile('instance')))
    for meme in memes:
      s = get_bs(f"{base_url}/{meme.get('href')}", parse_only=ss(class_='w100p'))
      caption = s.find('img')['alt']
      dfs.append(pd.DataFrame([[meme_href[1:], caption.split('-')[1][1:].lower()]], columns=['type', 'caption']))
    return pd.concat(dfs, ignore_index=True)
  except:
    return None

def main(n_pages_meme_types, n_pages, save_dir, outfile_name):
  dfs = []
  for j in range(n_pages):
    s = get_bs({'sort': 'top-2020-10', 'page': str(j + 1)}, parse_only=ss(class_='base-unit clearfix'))
    for img in s.find_all('img', class_='base-img'):
      raw_caption = img['alt']

  df = pd.concat(dfs, ignore_index=True)
  df.to_csv(f'{outfile_name}.csv')

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Helper script to download meme templates and captions from memegenerator.net')
  parser.add_argument('n_pages_meme_types', type=int, help='number of meme template pages to scrape')
  parser.add_argument('n_pages', type=int, help='number of memes per template to scrape')
  parser.add_argument('--save_dir', type=str, default='meme_templates', help='local directory to save scraped meme templates')
  parser.add_argument('--outfile_name', type=str, default='captions', help='CSV filename for scraped captions')
  args = parser.parse_args()
  main(args.n_pages_meme_types, args.n_pages, args.save_dir, args.outfile_name)
