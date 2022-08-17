
import xml.etree.ElementTree as ET
tree = ET.parse('/home/jinbu/text2mesh/data/scenenn/005-20220607T101740Z-001/005/005.xml')
root = tree.getroot()
for i in range(len(root)):
    data = root[i]
    label_id = root[i].attrib['id']
    color = root[i].attrib['color']
    label_text = root[i].attrib['text']
    nyu_class = root[i].attrib['nyu_class']
    print(f"{label_id}")

