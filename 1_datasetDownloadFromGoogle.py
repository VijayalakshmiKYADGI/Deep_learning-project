from simple_image_download import simple_image_download as sim

my_downloader = sim.Downloader()

my_downloader.directory = 'my_dir/'
# Change File extension type
my_downloader.extensions = '.jpg'
print(my_downloader.extensions)
my_downloader.download('Dallas_cowboys_football_players', limit=30)
#my_downloader.download('Dallas Cowboys sports players', limit=100)
#my_downloader.download('bananas', limit=100)
#my_downloader.download('mangoes', limit=100)
