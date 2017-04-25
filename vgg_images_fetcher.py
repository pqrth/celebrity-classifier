'''
Created on Feb 18, 2017
@author: Anantharaman
VGG Face dataset contains URLs and this tool visits these URLs and fetches the images
'''
import urllib
import os
from my_config.my_config import get_config

# CHANGE THE CODE BELOW TO YOUR PATHS
DS_PATH = get_config("DS_PATH")
VGG_PATH = os.path.join(DS_PATH, "vgg")
samples_path = os.path.join(VGG_PATH, "abridged_dataset") # we will keep some sample files here
myfiles = os.listdir(samples_path)
sample_files = [os.path.join(samples_path, f) for f in myfiles]
generated_images_path = os.path.join(VGG_PATH, "face_images")
#######################################
def fetch_urls(name, urls, destination):
    myext = ["jpg", "jpeg", "png"]
    
    fetched = 0
    for i, url in enumerate(urls):
        try:
            ext = url.rsplit("/", 1)[-1].split(".")[-1]
            ext = ext.lower()
            if ext not in myext:
                print ext, " NOT supported"
                continue
            fn = name + "_" + str(fetched) + "." + ext #url.rsplit("/", 1)[-1]
            fn = os.path.join(destination, fn)
            urllib.urlretrieve(url, fn)
            fetched += 1
        except Exception, e:
            print e 
            print "Error while fetching: ", url
        if (i % 10) == 0:
            print "i = %d, fetched = %d" % (i+1, fetched)
    return

def parse_ds_file(fn):
    result = []
    data = open(fn, "rb").readlines()
    for d in data:
        items = d.split()
        result.append({
         "name": fn.split(".")[0],
         "id": items[0], 
         "url": items[1], 
         "x1": items[2], "y1": items[3],  
         "x2": items[4], "y2": items[5], 
         "pose": items[6], 
         "det_score": items[7], 
         "curation": items[8], 
        })
    return result

def get_urls_from_files(sample_files):
    data = {}
    for fn in sample_files:
        urls = []
        result = parse_ds_file(fn)
        for res in result:
            if int(res["curation"]) == 1:
                urls.append(res["url"])
        name = os.path.split(fn)[1]
        data[name.split(".")[0]] = urls
    return data

if __name__ == '__main__':
    data = get_urls_from_files(sample_files)
    for name, urls in data.items():
        #print "gen path: ", generated_images_path, name
        mydir = os.path.join(generated_images_path, name)
        print "Generating Outputs to: ", mydir
        if not os.path.exists(mydir):
            os.makedirs(mydir)
        fetch_urls(name, urls, mydir)
