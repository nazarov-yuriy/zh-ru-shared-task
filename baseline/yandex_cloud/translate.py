import requests
import argparse
import os


ENDPOINT = "https://translate.api.cloud.yandex.net/translate/v2/translate"

def translate_texts(texts, folder_id, iam_token, dst_lang):
    data = {
        "folder_id": folder_id,
        "texts": texts,
        "targetLanguageCode": dst_lang,
    }
    headers = {'Authorization': 'Bearer ' + iam_token}
    r = requests.post(
        ENDPOINT,
        json=data,
        headers=headers
    )
    return [translation["text"] for translation in r.json()["translations"]]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--src")
    argparser.add_argument("--dst")
    argparser.add_argument("--dst-lang", default="ru")
    argparser.add_argument("--folder_id")
    args = argparser.parse_args()
    iam_token = os.environ["IAM_TOKEN"]
    with open(args.src, "r") as src_file, open(args.dst, "w") as dst_file:
        texts = []
        for text in src_file:
            texts.append(text.rstrip())
        for text in translate_texts(texts, folder_id=args.folder_id, iam_token=iam_token, dst_lang=args.dst_lang):
            print(text, file=dst_file)
