import xml.etree.ElementTree as ET

files = ['train.txt', 'val.txt']

for file in files:
    image_ids = list()
    f = open(f'dataset/soda10m_aspascal/ImageSets/Main/{file}')
    for line in f:
        image_ids.append(line.rstrip())

    ids_to_remove = list()
    for image_id in image_ids:
        annotation_file = f'dataset/soda10m_aspascal/Annotations/{image_id}.xml'
        objects = ET.parse(annotation_file).findall("object")

        if len(objects) == 0:
            ids_to_remove.append(image_id)

    outfile = open(f'dataset/soda10m_aspascal/ImageSets/Main/{file}',
                   'w')

    for image_id in image_ids:
        if image_id not in ids_to_remove:
            outfile.write(image_id + '\n')

    print(f'{len(ids_to_remove)} items removed from {file}')
