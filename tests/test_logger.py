import importlib
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class LoggerSetupTests(unittest.TestCase):
    def setUp(self):
        self.logger_module = importlib.import_module('src.logger')
        self.original_log_to_file = os.environ.pop('LOG_TO_FILE', None)
        self.original_log_level = os.environ.get('LOG_LEVEL')
        os.environ['LOG_LEVEL'] = 'INFO'
        self.original_logs_dir = self.logger_module.LOGS_DIR
        self.loggers_to_cleanup = []

    def tearDown(self):
        if self.original_log_to_file is None:
            os.environ.pop('LOG_TO_FILE', None)
        else:
            os.environ['LOG_TO_FILE'] = self.original_log_to_file

        if self.original_log_level is None:
            os.environ.pop('LOG_LEVEL', None)
        else:
            os.environ['LOG_LEVEL'] = self.original_log_level

        self.logger_module.LOGS_DIR = self.original_logs_dir
        for logger_name in self.loggers_to_cleanup:
            logger = logging.getLogger(logger_name)
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()

    def setup_test_logger(self, name: str) -> logging.Logger:
        self.loggers_to_cleanup.append(name)
        return self.logger_module.setup_logger(name)

    def test_import_does_not_create_logs_directory(self):
        logs_dir = REPO_ROOT / 'logs'
        if logs_dir.exists():
            self.skipTest('logs directory already exists')

        subprocess.run(
            [sys.executable, '-c', 'import src.logger'],
            cwd=REPO_ROOT,
            env={**os.environ, 'LOG_TO_FILE': ''},
            check=True,
        )

        self.assertFalse(logs_dir.exists())

    def test_setup_logger_uses_console_only_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / 'logs'
            self.logger_module.LOGS_DIR = logs_dir

            logger = self.setup_test_logger('test_console_only_logger')

            self.assertTrue(any(handler.name == 'margo_console' for handler in logger.handlers))
            self.assertFalse(any(isinstance(handler, logging.FileHandler) for handler in logger.handlers))
            self.assertFalse(logs_dir.exists())

    def test_setup_logger_adds_file_handlers_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / 'logs'
            self.logger_module.LOGS_DIR = logs_dir
            os.environ['LOG_TO_FILE'] = 'true'

            logger = self.setup_test_logger('test_file_logger')
            file_handlers = [
                handler
                for handler in logger.handlers
                if isinstance(handler, logging.FileHandler)
            ]

            self.assertEqual(2, len(file_handlers))
            self.assertTrue(logs_dir.exists())
            self.assertTrue(any(path.name.startswith('translator_') for path in logs_dir.iterdir()))
            self.assertTrue(any(path.name.startswith('errors_') for path in logs_dir.iterdir()))

    def test_setup_logger_can_add_file_handlers_after_console_setup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / 'logs'
            self.logger_module.LOGS_DIR = logs_dir
            logger_name = 'test_file_logger_after_console'

            logger = self.setup_test_logger(logger_name)
            self.assertFalse(any(isinstance(handler, logging.FileHandler) for handler in logger.handlers))

            os.environ['LOG_TO_FILE'] = 'true'
            logger = self.logger_module.setup_logger(logger_name)
            file_handlers = [
                handler
                for handler in logger.handlers
                if isinstance(handler, logging.FileHandler)
            ]

            self.assertEqual(2, len(file_handlers))
            self.assertTrue(logs_dir.exists())

    def test_setup_logger_removes_file_handlers_when_disabled_after_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / 'logs'
            self.logger_module.LOGS_DIR = logs_dir
            logger_name = 'test_file_logger_true_to_false'
            os.environ['LOG_TO_FILE'] = 'true'

            logger = self.setup_test_logger(logger_name)
            file_handlers = [
                handler
                for handler in logger.handlers
                if isinstance(handler, logging.FileHandler)
            ]
            self.assertEqual(2, len(file_handlers))

            os.environ['LOG_TO_FILE'] = 'false'
            logger = self.logger_module.setup_logger(logger_name)

            self.assertTrue(any(handler.name == 'margo_console' for handler in logger.handlers))
            self.assertFalse(any(isinstance(handler, logging.FileHandler) for handler in logger.handlers))
            self.assertTrue(all(handler.stream is None for handler in file_handlers))

    def test_truncated_suffix_requires_debug_capable_file_logging(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / 'logs'
            self.logger_module.LOGS_DIR = logs_dir
            os.environ['LOG_TO_FILE'] = 'true'
            os.environ['LOG_LEVEL'] = 'INFO'

            logger = self.setup_test_logger('test_info_file_suffix')

            self.assertEqual(
                '[truncated, set LOG_LEVEL=DEBUG for full prompt file logs]',
                self.logger_module._truncated_suffix('prompt', logger),
            )

    def test_truncated_suffix_points_to_log_file_when_debug_reaches_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / 'logs'
            self.logger_module.LOGS_DIR = logs_dir
            os.environ['LOG_TO_FILE'] = 'true'
            os.environ['LOG_LEVEL'] = 'DEBUG'

            logger = self.setup_test_logger('test_debug_file_suffix')

            self.assertEqual(
                '[truncated, see log file for full response]',
                self.logger_module._truncated_suffix('response', logger),
            )


if __name__ == '__main__':
    unittest.main()
