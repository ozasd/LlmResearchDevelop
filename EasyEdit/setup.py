from setuptools import setup, find_packages

setup(
    name="easyeditor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "easyeditor",  # 順便把剛剛缺的這個補進去
        # 其他依賴已經在 requirements.txt 裡了，這裡只要確保結構正確
    ],
)