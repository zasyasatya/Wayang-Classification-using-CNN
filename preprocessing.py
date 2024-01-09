import shutil
import os
import cv2

# path = "D:/dataset/DatasetWayang/A_Data_Test"
path = 'D:/dataset/astungkara/cleanVersion_080522_1080/'
dest_path = 'D:/dataset/astungkara/cleanVersion_080522_1080/'
wayangClass = ["Abimanyu", "Acintya",
"AnantaBhoga",
"Api",
"Arjuna",
"Aswatama",
"Baladewa",
"Basudewa",
"Bayu",
"Biasa",
"Bima",
"Bisma",
"Brahma",
"Condong",
"Delem",
"Drestadyumna",
"Drona",
"Drupada",
"Drupadi",
"Durga",
"Dursasana",
"Duryodana",
"Dwala",
"Ganesha",
"Garuda",
"Gatotkaca",
"Hanoman",
"Hidimba",
"Indra",
"Jayadrata",
"Jogormanik",
"KalaDremba",
"Kanwa",
"Karna",
"Krisna",
"Kumbakarna",
"Kunti",
"Laksamana",
"LudraMurti",
"Meganada",
"Merdah",
"Nakula",
"Nala",
"Narada",
"Pandu",
"Prajurit",
"Rahwana",
"Rama",
"Rangda",
"Sahadewa",
"Sakuni",
"Salya",
"Sangut",
"Saraswati",
"Satyaki",
"Sita",
"Siwa",
"Sugriwa",
"Suratma",
"Suweta",
"Tualen",
"Wibisana",
"Widura",
"WisnuMurti",
"Yudistira",
"Yuyutsu"]


# for items in wayangClass:
#     # print(path + '/' + items)
#     # os.mkdir(path + '/' + items)
#     finalPath = os.path.join(path, items)
#     # print(finalPath)
#     os.mkdir(finalPath)

import os

path = 'D:/KULIAH/SKRIPSI/dataset/v1/gt/'
dest_path = 'D:/KULIAH/SKRIPSI/dataset/v1/gt/'

# Rename File with Ordered List
for i, ret in enumerate(os.walk(path)):
  for i, filename in enumerate(sorted(ret[2], key = len)):
      source = path + filename
      destination = dest_path + 'gt' + '_' + str(i) + '.png'
      os.rename(source, destination)
      # print(i)
      
      # newImage = cv2.imread(path + filename)
      # newImage = cv2.resize(newImage, (1920, 1080))
      # cv2.imwrite(os.path.join(dest_path , filename), newImage)
      # # i+=1
      if filename.startswith("."):
        continue
      
      print(filename)
      # for directory in wayangClass:
      #     if (splited == directory):
      #         src = os.path.join(path, filename)
      #         dst = os.path.join(path, directory)
      #         # print(src)
      #         print(splited)
      #         # shutil.copyfile(src, dst)
      
      