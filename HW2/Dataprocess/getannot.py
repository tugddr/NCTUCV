#coding=utf-8
import os
import shutil
from random import sample

def copyFiles(sourceDir,targetDir):
  if sourceDir.find("exceptionfolder")>0:
    return
  i = 0
  for file in os.listdir(targetDir):
    target = file[:-4]
    print(target)
    sourceFile = os.path.join(sourceDir,target+".txt")
    targetFile = os.path.join(targetDir,target+".txt")
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

  
if __name__ =="__main__":
  copyFiles("yolocoords","data/valid")