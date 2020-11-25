# Adapted from: https://github.com/ruhomor/Meme-Generator/blob/master/data_scraper.py

# Keep imports lightweight
from bs4 import BeautifulSoup as bs
from bs4 import SoupStrainer as ss
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import path, makedirs
from requests import get as requests_get
from pandas import DataFrame, concat
from re import compile as re_compile
from re import DOTALL
from sys import stderr

BASE_URL = 'https://imgflip.com'
MEME_TEMPLATES_URL = f'{BASE_URL}/memetemplates'

RE_0 = re_compile(r'Blank Meme Template')
RE_1 = re_compile(r'\| image tagged in memes,.*\| made w/ Imgflip meme maker')
RE_1_SPAN_OFFSETS = (24, 29)
RE_2 = re_compile(r'\|.*?\|', flags=DOTALL)  # Make sure a newline character in caption does not mess up regex
RE_2_SPAN_OFFSETS = (3, 2)

def get_bs(url, payload=None, parse_only=None):
  r = requests_get(url, params=payload)  # Get HTML source as string
  return bs(r.text, 'html.parser', parse_only=parse_only)  # parse_only: pass SoupStrainer to filter out HTML elements while parsing

def get_meme_template_info(meme_template):
  link = meme_template.a
  return (link['href'], link.text)

def save_meme_template(meme_href, save_dir):
  # Follow that URL to get full-resolution meme template image
  s = get_bs(f'{BASE_URL}{meme_href}', parse_only=ss('img', alt=RE_0))
  image_url = s.find('img')['src']
  # Download image
  image = requests_get(f'{BASE_URL}{image_url}', stream=True).content
  image_name = image_url.split('/')[-1]
  with open(f'{save_dir}/{image_name}', 'wb') as outfile:
    outfile.write(image)

def get_meme_info(page, meme_href, meme_name):
  dfs = []
  memes = get_bs(f'{BASE_URL}{meme_href}', payload={'page': page}, parse_only=ss(class_='base-unit clearfix'))
  # Break early as needed
  if len(memes) == 0:
    return None
  else:
    for meme in memes:
      img = meme.find('img', class_='base-img')
      # Make sure meme is actually an image (rather than GIF)
      if img is not None:
        alt_text = img['alt']
        # Make sure meme actually has a caption
        if alt_text.count('|') == 3:
          view_info = meme.find(class_='base-view-count').text.replace(',', '').split()
          tags_search = RE_1.search(alt_text)
          caption_search = RE_2.search(alt_text)
          # Make sure meme has metadata containing number of views and upvotes (OK if there are no comments)
          if len(view_info) >= 4 and tags_search is not None and caption_search is not None:
            views, upvotes = int(view_info[0]), int(view_info[2])
            tags_span, caption_span = tags_search.span(), caption_search.span()
            tags = alt_text[tags_span[0] + RE_1_SPAN_OFFSETS[0]:tags_span[1] - RE_1_SPAN_OFFSETS[1]]
            caption = alt_text[caption_span[0] + RE_2_SPAN_OFFSETS[0]: caption_span[1] - RE_2_SPAN_OFFSETS[1]].replace('; ', '\n')
            dfs.append(DataFrame([[meme_name, caption, tags, views, upvotes]], columns=['type', 'caption', 'tags', 'views', 'upvotes']))
    return None if len(dfs) == 0 else concat(dfs, ignore_index=True)

def process_meme_template(meme_template, save_dir, n_pages_per_meme, num_attempts):
  meme_href, meme_name = get_meme_template_info(meme_template)
  save_meme_template(meme_href, save_dir)
  dfs = []
  for i in range(n_pages_per_meme):
    for attempt in range(num_attempts):
      try:
        result = get_meme_info(str(i + 1), meme_href, meme_name)
        if result is not None:
          dfs.append(result)
      except Exception as e:
        print(f'Exception occurred during attempt {attempt} for page {i} and meme template {meme_template}.', file=stderr)
        print(e, file=stderr)
      else:
        break
    else:
      # All attempts failed
      print(f'Attempted {num_attempts}, but all of them failed.', file=stderr)
  return None if len(dfs) == 0 else concat(dfs, ignore_index=True)

def process_meme_templates(page, save_dir, n_pages_per_meme, num_attempts):
  dfs = []
  with ThreadPoolExecutor() as executor:
    futures = [executor.submit(lambda x: process_meme_template(x, save_dir, n_pages_per_meme, num_attempts), meme_template) for meme_template in get_bs(MEME_TEMPLATES_URL, payload={'page': str(page + 1)}, parse_only=ss(class_='mt-title'))]
    for future in as_completed(futures):
      result = future.result()
      if result is not None:
        dfs.append(result)
  return None if len(dfs) == 0 else concat(dfs, ignore_index=True)

def main(n_pages_meme_types, n_pages_per_meme, save_dir, outfile, num_attempts):
  # Create save_dir folder if it doesn't already exist
  if not path.exists(save_dir):
    makedirs(save_dir)
  # Scrape driver
  dfs = []
  for i in range(n_pages_meme_types):
    dfs.append(process_meme_templates(i, save_dir, n_pages_per_meme, num_attempts))
  if len(dfs) == 0:
    print('No valid captions were found for the given parameters.', file=stderr)
  else:
    df = concat(dfs, ignore_index=True)
    print(f'# of memes scraped: {len(df)}')
    df.to_csv(f'{outfile}.tsv', sep='\t')

if __name__ == "__main__":
  parser = ArgumentParser(description="Meme caption and metadata curation script for imgflip. Note that the specified folder for saving the meme templates will be created if it does not already exist. In captions, the sequence '; ' is converted to '\n' to normalize the formatting convention of imgflip.", formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument('n_pages_meme_types', type=int, help='number of meme template pages to scrape')  # pages of meme templates
  parser.add_argument('n_pages_per_meme', type=int, help='number of memes per template to scrape')  # pages of memes per template
  parser.add_argument('-d', '--save_dir', type=str, default='meme_templates', help='local directory to save meme templates for which captions were found')
  parser.add_argument('-o', '--outfile', type=str, default='captions', help='TSV filename (without extension) for scraped captions and metadata')
  parser.add_argument('-a', '--num_attempts', type=int, default=10, help='number of times to retry processing a meme template if an exception is thrown')
  args = parser.parse_args()
  main(args.n_pages_meme_types, args.n_pages_per_meme, args.save_dir, args.outfile, args.num_attempts)
