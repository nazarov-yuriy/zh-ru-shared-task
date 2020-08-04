import argparse
from google.cloud import translate_v2 as translate


def translate_texts(texts, project_id, dst_lang):
    translate_client = translate.Client()

    results = translate_client.translate(texts, target_language=dst_lang)
    return [result['translatedText'] for result in results]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--src")
    argparser.add_argument("--dst")
    argparser.add_argument("--dst-lang", default="ru")
    argparser.add_argument("--project_id")
    args = argparser.parse_args()
    with open(args.src, "r") as src_file, open(args.dst, "w") as dst_file:
        texts = []
        for text in src_file:
            texts.append(text.rstrip())
        for text in translate_texts(texts, project_id=args.project_id, dst_lang=args.dst_lang):
            print(text, file=dst_file)
