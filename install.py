import subprocess, sys

def install():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'HDRutils', 'imageio[freeimage]'])

if __name__ == '__main__':
    install()
