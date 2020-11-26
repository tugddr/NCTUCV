#coding=utf-8
import os
import shutil
from random import sample

def removeFileInDir(sourceDir,files):
  i = 0
  for file in files:
    i+=1
    file=os.path.join(sourceDir,file) #必須拼接完整檔名
    if os.path.isfile(file) and file.find(".png")>0:
      os.remove(file)
      print(file+" remove succeeded")
  print(i)


def copyFiles(sourceDir,targetDir,valid_is_already):
  if sourceDir.find("exceptionfolder")>0:
    return
  files = []
  valid_files = []
  if not valid_is_already:
    for file in os.listdir(sourceDir):
      files.append(file)
    #valid_files = sample(files,6000)
  else:
    with open('data/valid.txt', 'r') as files:
      for line in files:
        file = line[11:-1]
        valid_files.append(file)
    files.close()
    
  i = 0
  for file in valid_files:
    sourceFile = os.path.join(sourceDir,file)
    targetFile = os.path.join(targetDir,file)
    i+=1
    if os.path.isfile(sourceFile):
      if not os.path.exists(targetDir):
        os.makedirs(targetDir)

      if not os.path.exists(targetFile) or (os.path.exists(targetFile) and (os.path.getsize(targetFile) != os.path.getsize(sourceFile))):
        open(targetFile, "wb").write(open(sourceFile, "rb").read())
        print(targetFile+" copy succeeded")

    if os.path.isdir(sourceFile):
      copyFiles(sourceFile, targetFile)
  print(i)
  removeFileInDir(sourceDir,valid_files)
  

  
if __name__ =="__main__":
  valid_is_already = True
  
  copyFiles("data/obj","data/valid",valid_is_already)