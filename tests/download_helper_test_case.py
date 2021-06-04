# download_helper_test_case.py

import os
import shutil
import unittest

from pathlib import Path
from src.download_helper import DownloadHelper


class DownloadHelperTestCase(unittest.TestCase):
    """Test names are generally self-explanatory and so docstrings not provided on an individual basis other
    than by exception.

    Keyword arguments:
    TestCase -- standard class required for tests based on unittest.case
    """

    def setUp(self):
        """Fixtures used by tests."""
        self.Root = Path(__file__).parent
        self.FakePath = os.path.join(self.Root, "FakeTestFile.zip")
        self.RealFilePath = os.path.join(self.Root, "TestFile.txt")
        self.TestDestinationFolder = os.path.join(self.Root, "test_folder")
        self.TestDestinationFile = os.path.join(self.TestDestinationFolder, "TestFile.txt")

    def test_can_download__when_target_is_not_file__returns_true(self):
        test_file = self.FakePath

        with self.subTest(self):
            self.assertTrue(DownloadHelper.can_download(target_path=test_file))
            self.assertTrue(DownloadHelper.can_download(target_path=test_file, replace_download='n'))
            self.assertTrue(DownloadHelper.can_download(target_path=test_file, replace_download='y'))

    def test_can_download__when_target_is_file__uses_replace_download(self):
        test_file = self.RealFilePath

        # Not testing user input route - would need to fake it
        with self.subTest(self):
            self.assertFalse(DownloadHelper.can_download(target_path=test_file, replace_download='n'))
            self.assertTrue(DownloadHelper.can_download(target_path=test_file, replace_download='y'))

    def test_can_extract_to_extraction_dir__when_target_is_not_dir__returns_true(self):
        test_path = self.FakePath

        with self.subTest(self):
            self.assertTrue(DownloadHelper.can_extract_to_extraction_dir(unzip_dir=test_path))
            self.assertTrue(DownloadHelper.can_extract_to_extraction_dir(unzip_dir=test_path,
                                                                      replace_content='n'))
            self.assertTrue(DownloadHelper.can_extract_to_extraction_dir(unzip_dir=test_path,
                                                                      replace_content='y'))

    def test_can_extract_to_extraction_dir__when_empty_target_exists__uses_replace_destination_dir(self):
        test_path = self.TestDestinationFolder

        # Not testing user input route - would need to fake it
        with self.subTest(self):
            # Start with empty test dir
            self.assertTrue(DownloadHelper.can_extract_to_extraction_dir(unzip_dir=test_path,
                                                                      replace_content='n'))
            self.assertTrue(DownloadHelper.can_extract_to_extraction_dir(unzip_dir=test_path,
                                                                      replace_content='y'))

    def test_can_extract_to_extraction_dir__when_non_empty_target_exists__uses_replace_destination_dir(self):
        test_path = self.TestDestinationFolder
        # Add temp file
        shutil.copy(self.RealFilePath, self.TestDestinationFolder)

        # Not testing user input route - would need to fake it
        with self.subTest(self):
            # Start with empty test dir
            self.assertFalse(DownloadHelper.can_extract_to_extraction_dir(unzip_dir=test_path,
                                                                       replace_content='n'))
            self.assertTrue(DownloadHelper.can_extract_to_extraction_dir(unzip_dir=test_path,
                                                                      replace_content='y'))

        # Clean up - remove test file
        os.remove(self.TestDestinationFile)

    def test_get_extraction_dir_path__when_known_file_endswith__returns_path(self):
        test_path = self.TestDestinationFolder
        dir_name = 'training_input'
        filename = 'dummy' + dir_name + '.zip'
        filepath = Path(DownloadHelper.get_extraction_dir_path(test_path, filename))
        self.assertTrue(filepath.name == dir_name)

    def test_get_extraction_dir_path__when_unknown_file_endswith__returns_empty_string(self):
        test_path = self.TestDestinationFolder
        dir_name = 'fred'
        filename = 'dummy' + dir_name + '.zip'
        filepath = Path(DownloadHelper.get_extraction_dir_path(test_path, filename))
        self.assertTrue(filepath.name == '')


if __name__ == '__main__':
    unittest.main()
