import requests
import pandas as pd
from PIL import Image

def getImage(url, name):
  with open(name, "wb") as img:
    response = requests.get(url, stream=True)
    if not response.ok:
      print(response)
    for block in response.iter_content(1024):
      if not block: break
      img.write(block)
    img.close()
  
  im = Image.open(name)
  im = im.resize((256, 256), Image.LANCZOS)
  im.save(name)

df = pd.read_csv("photos.csv")
df_subset = df[:][7351:7351+120]

print("running")
for url, name in zip(list(df_subset["photo_image_url"]), list(df_subset["photo_id"])):
  getImage(url, "photos/test/test_data/" + name+".jpg")
