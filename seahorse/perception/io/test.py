
from screenshot_source import ScreenshotSource

def test_screenshot_source():
    source = "source 0"
    screen_shots = ScreenshotSource(source)
    for s in screen_shots:
        print(s)


if __name__ == "__main__":
    test_screenshot_source()
