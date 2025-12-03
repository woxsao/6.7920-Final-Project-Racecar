from PIL import Image

'''
The map png must be greyscale format, NOT RGB
'''

PATH_TO_PNG = '2_rectangle/2_rectangle_map.png'

img = Image.open(PATH_TO_PNG)
print(img.mode)
img = img.convert('L')
img.save(PATH_TO_PNG)

img = Image.open(PATH_TO_PNG)
print(img.mode)
assert img.mode == 'L'