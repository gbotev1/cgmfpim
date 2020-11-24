# Adapted from: https://github.com/ruhomor/Meme-Generator/blob/master/data_scraper.py

# Keep imports lightweight
from bs4 import BeautifulSoup as bs
from bs4 import SoupStrainer as ss
from concurrent.futures import ThreadPoolExecutor, as_completed
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from os import path, makedirs
from re import compile as re_compile
from requests import get as requests_get
from pandas import DataFrame, concat

BASE_URL = 'https://imgflip.com'
MEME_TEMPLATES_URL = f'{BASE_URL}/memetemplates'

RE_0 = re_compile('Blank Meme Template')

def get_bs(url, payload=None, parse_only=None):
  r = requests_get(url, params=payload) # full text of HTML file
  return bs(r.text, 'html.parser', parse_only=parse_only) # parse_only: what elements to get from HTML file

def save_meme_template(meme_template, save_dir):
  # Create save_dir folder if it doesn't already exist
  if not path.exists(save_dir):
    makedirs(save_dir)
  # Get local meme template URL
  meme_href = meme_template.a['href']
  # Follow that URL to get full-resolution meme template image
  s = get_bs(f'{BASE_URL}{meme_href}', parse_only=ss('img', alt=RE_0))
  image_url = s.find('img')['src']
  # Download image
  image = requests_get(f"{BASE_URL}{image_url}", stream=True).content
  image_name = image_url.split('/')[-1]
  with open(f'{save_dir}/{image_name}', 'wb') as outfile:
    outfile.write(image)

def get_meme_info(meme_template):
  try:
    dfs = []
    # get local meme template URL and name
    meme_href = meme_template.a['href']
    meme_name = meme_template.a.string
    s = get_bs(f'{BASE_URL}{meme_href}', parse_only=ss(class_='base-unit clearfix'))

    # for each meme in template:
    for item in s:
      img = item.find('img', class_="base-img")
      view_info = item.find(class_="base-view-count")
      # if item is an image (rather than video)
      if img is not None:
        # get caption and tags
        raw_caption = img['alt'].split('|')
        caption = ""
        tags = []
        for i, item in enumerate(raw_caption):
          if item.startswith(" image tagged in"):
            tags = item.split(" image tagged in")[-1].strip().split(',')
            if raw_caption[i-1].startswith('  '):
              caption = raw_caption[i-1].strip()

        # get views and upvotes
        meme_stat_list = [x.strip(',') for x in view_info.string.strip().split()]
        # not sure if the following conditional is necessary; it checks that both views and upvotes are included
        # if all(stat in meme_stat_list for stat in ["views", "upvotes"]):
        views = int(meme_stat_list[0].replace(',',''))
        upvotes = int(meme_stat_list[2].replace(',',''))
        
        # append all of this information (name, caption, tags, views, upvotes) into the dataframe
        dfs.append(DataFrame([[meme_name, caption, tags, views, upvotes]], columns=['type', 'caption', 'tags', 'view_count', 'upvotes']))
    
    return concat(dfs, ignore_index=True)
  except:
    return None

def main(n_pages_meme_types, n_pages_per_meme, save_dir, outfile):
  dfs = []
  for page in range(n_pages_meme_types):
    for meme_template in get_bs(MEME_TEMPLATES_URL, payload={'page': str(page + 1)}, parse_only=ss(class_='mt-title')):
      save_meme_template(meme_template, save_dir)
      dfs.append(get_meme_info(meme_template))
  df = concat(dfs, ignore_index=True)
  df.to_csv(f'{outfile}.tsv', sep='\t')

if __name__ == "__main__":
  parser = ArgumentParser(description='Meme caption and metadata curation script for imgflip. Note that the specified folder for saving the meme templates will be created if it does not already exist.', formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument('n_pages_meme_types', type=int, help='number of meme template pages to scrape') # pages of memes
  parser.add_argument('n_pages_per_meme', type=int, help='number of memes per template to scrape') # pages of memes per template
  parser.add_argument('-d', '--save_dir', type=str, default='meme_templates', help='local directory to save meme templates for which captions were found')
  parser.add_argument('-o', '--outfile', type=str, default='captions', help='TSV filename (without extension) for scraped captions and metadata')
  args = parser.parse_args()
  main(args.n_pages_meme_types, args.n_pages_per_meme, args.save_dir, args.outfile)
