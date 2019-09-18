import argparse
import requests
import sys

def validate(args):
    if args.get == args.post:
        sys.exit("pick get or post, not both")

    if not args.api:
        sys.exit("pick an API")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--get", action="store_true", dest="get")
    parser.add_argument("--post", action="store_true", dest="post")
    parser.add_argument("--api", action="store", dest="api", default=None)
    parser.add_argument("--image-path", action="store", dest="image", default=None)
    args = parser.parse_args()
    validate(args)

    return args.get, args.post, args.api, args.image

def main():
    get, post, api, image = get_args()

    host  = '127.0.0.1'
    port = 5000
    url  = "http://{}:{}/{}".format(host, port, api)

    if get:
        r = requests.get(url=url)

    if post:
        files = {'image' : open(image, 'rb')}
        r     = requests.post(url=url, files=files)

    #print(dir(r))
    print(r.status_code)
    print(r.text)
    print(r.json())

if __name__ == "__main__":
    main()
