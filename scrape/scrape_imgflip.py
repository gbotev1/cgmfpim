# Adapted from: https://github.com/ruhomor/Meme-Generator/blob/master/data_scraper.py

# Keep imports lightweight
from bs4 import BeautifulSoup as bs
from bs4 import SoupStrainer as ss
from concurrent.futures import ThreadPoolExecutor, as_completed
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from os import path, makedirs
from requests import get as requests_get
from pandas import DataFrame, concat
from re import compile as re_compile
from sys import stderr

BASE_URL = 'https://imgflip.com'
MEME_TEMPLATES_URL = f'{BASE_URL}/memetemplates'

RE_0 = re_compile(r'Blank Meme Template')
RE_1 = re_compile(r'\| image tagged in memes,.*\| made w/ Imgflip meme maker')
RE_1_SPAN_OFFSETS = (24, 29)
RE_2 = re_compile(r'\|.*?\|')
RE_2_SPAN_OFFSETS = (3, 2)

def get_bs(url, payload=None, parse_only=None):
  r = requests_get(url, params=payload)  # Get HTML source as string
  return bs(r.text, 'html.parser', parse_only=parse_only)  # parse_only: pass SoupStrainer to filter out HTML elements while parsing

def get_meme_template_info(meme_template):
  link = meme_template.a
  return (link['href'], link.text)

def save_meme_template(meme_href, save_dir):
  # Create save_dir folder if it doesn't already exist
  if not path.exists(save_dir):
    makedirs(save_dir)
  # Follow that URL to get full-resolution meme template image
  s = get_bs(f'{BASE_URL}{meme_href}', parse_only=ss('img', alt=RE_0))
  image_url = s.find('img')['src']
  # Download image
  image = requests_get(f"{BASE_URL}{image_url}", stream=True).content
  image_name = image_url.split('/')[-1]
  with open(f'{save_dir}/{image_name}', 'wb') as outfile:
    outfile.write(image)

def get_meme_info(meme_href, meme_name):
  dfs = []
  memes = get_bs(f'{BASE_URL}{meme_href}', parse_only=ss(class_='base-unit clearfix'))
  for meme in memes:
    try:
      img = meme.find('img', class_='base-img')
      # Make sure meme is actually an image (rather than GIF)
      if img is not None:
        # Make sure there is a caption
        alt_text = img['alt']
        if alt_text.count('|') == 3:
          # View info
          view_info = meme.find(class_='base-view-count')
          views, upvotes, comments = [int(item.replace(',', '')) for i, item in enumerate(view_info.text.split()) if i & 1 == 0]
          # Tags
          tags_span = RE_1.search(alt_text).span()
          tags = alt_text[tags_span[0] + RE_1_SPAN_OFFSETS[0]:tags_span[1] - RE_1_SPAN_OFFSETS[1]]
          # Caption
          caption_span = RE_2.search(alt_text).span()
          caption = alt_text[caption_span[0] + RE_2_SPAN_OFFSETS[0]: caption_span[1] - RE_2_SPAN_OFFSETS[1]]
          # Partial write
          dfs.append(DataFrame([[meme_name, caption, tags, views, upvotes, comments]], columns=['type', 'caption', 'tags', 'views', 'upvotes', 'comments']))
    except:
      pass
  return None if len(dfs) == 0 else concat(dfs, ignore_index=True)

def process_template(meme_template, save_dir):
  meme_href, meme_name = get_meme_template_info(meme_template)
  save_meme_template(meme_href, save_dir)
  return get_meme_info(meme_href, meme_name)

def process_page(page, save_dir):
  dfs = []
  for meme_template in get_bs(MEME_TEMPLATES_URL, payload={'page': str(page + 1)}, parse_only=ss(class_='mt-title')):
    result = process_template(meme_template, save_dir)
    if result is not None:
      dfs.append(result)
  return None if len(dfs) == 0 else concat(dfs, ignore_index=True)

def main(n_pages_meme_types, n_pages_per_meme, save_dir, outfile):
  dfs = []
  with ThreadPoolExecutor() as executor:
    futures = [executor.submit(lambda i: process_page(i, save_dir), i) for i in range(n_pages_meme_types)]
    for future in as_completed(futures):
      future_result = future.result()
      if future_result is not None:
        dfs.append(future_result)
  if len(dfs) == 0:
    print('No valid captions were found for the given parameters.', file=stderr)
  else:
    df = concat(dfs, ignore_index=True)
    df.to_csv(f'{outfile}.tsv', sep='\t')

if __name__ == "__main__":
  parser = ArgumentParser(description='Meme caption and metadata curation script for imgflip. Note that the specified folder for saving the meme templates will be created if it does not already exist.', formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument('n_pages_meme_types', type=int, help='number of meme template pages to scrape')  # pages of memes
  parser.add_argument('n_pages_per_meme', type=int, help='number of memes per template to scrape')  # pages of memes per template
  parser.add_argument('-d', '--save_dir', type=str, default='meme_templates', help='local directory to save meme templates for which captions were found')
  parser.add_argument('-o', '--outfile', type=str, default='captions', help='TSV filename (without extension) for scraped captions and metadata')
  args = parser.parse_args()
  main(args.n_pages_meme_types, args.n_pages_per_meme, args.save_dir, args.outfile)
