# -*- coding: utf-8 -*-
"""
Entry point for the Auto Clip Refiner application.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__" and __package__ is None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

from PySide6.QtWidgets import QApplication

from scripts.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
