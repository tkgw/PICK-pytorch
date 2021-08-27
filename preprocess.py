import csv
import json
import os
import shutil

import pandas
from sklearn.model_selection import train_test_split

# Input dataset
data_path = 'ICDAR-2019-SROIE/data/'
box_path = data_path + 'box/'
img_path = data_path + 'img/'
key_path = data_path + 'key/'

# Output dataset
out_boxes_and_transcripts = 'data/data_examples_root/boxes_and_transcripts/'
out_images = 'data/data_examples_root/images/'
out_entities = 'data/data_examples_root/entities/'

train_samples_list = []
for file in os.listdir(box_path):

    # Reading csv
    with open(box_path + file, 'r') as fp:
        reader = csv.reader(fp, delimiter=',')
        # arranging dataframe index ,coordinates x1_1,y1_1,x2_1,y2_1,x3_1,y3_1,x4_1,y4_1, transcript
        rows = [[1] + x[:8] + [','.join(x[8:]).strip(',')] for x in reader]
        df = pandas.DataFrame(rows)

    # including ner label dataframe index ,coordinates x1_1,y1_1,x2_1,y2_1,x3_1,y3_1,x4_1,y4_1, transcript , ner tag
    df[10] = 'other'

    # saving file into new dataset folder
    jpg = file.replace('.csv', '.jpg')
    with open(key_path + file.replace('.csv', '.json')) as fp:
        entities = json.load(fp)
    for key, value in sorted(entities.items()):
        idx = df[df[9].str.contains('|'.join(map(str.strip, value.split(','))))].index
        df.loc[idx, 10] = key

    shutil.copy(img_path + jpg, out_images)
    with open(out_entities + file.replace('.csv', '.txt'), 'w') as fp:
        json.dump(entities, fp=fp)

    df.to_csv(out_boxes_and_transcripts + file.replace('.csv', '.tsv'), index=False, header=False, quotechar='', escapechar='\\', quoting=csv.QUOTE_NONE, )
    train_samples_list.append(['receipt', file.replace('.csv', '')])
train_samples_list = pandas.DataFrame(train_samples_list, columns=['document_type', 'file_name'])

train, test = train_test_split(train_samples_list, test_size=0.2, random_state=42)

train.reset_index(drop=True, inplace=True)
train.to_csv('data/data_examples_root/train_samples_list.csv', header=False)


def move(src, dst):
    shutil.copy(src, dst)
    os.remove(src)


for file in test.file_name:
    move(out_boxes_and_transcripts + file + '.tsv', 'data/test_data_example/boxes_and_transcripts/')
    move(out_images + file + '.jpg', 'data/test_data_example/images/')
    move(out_entities + file + '.txt', 'data/test_data_example/entities/')

test.reset_index(drop=True, inplace=True)
test.to_csv('data/test_data_example/test_samples_list.csv', header=False)
