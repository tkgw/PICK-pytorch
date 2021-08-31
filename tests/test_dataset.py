# import unittest

from pick.data_utils.pick_dataset import PICKDataset


# class TestDataset(unittest.TestCase):
#     def test_health(self):
#         pass


if __name__ == '__main__':

    dataset = PICKDataset(
        files_name='data/data_examples_root/train_samples_list.csv',
        boxes_and_transcripts_folder='boxes_and_transcripts',
        images_folder='images',
        entities_folder='entities',
        iob_tagging_type='box_and_within_box_level',
        resized_image_size=(480, 960),
        ignore_error=False)
