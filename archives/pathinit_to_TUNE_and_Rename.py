import os
import sys

rootPath_to_JaxTutos_localrepository = '/sps/lsst/users/campagne/'
rootPath_to_JaxTutos_conda_env = '/sps/lsst/users/campagne/anaconda3/envs/'
homePath = os.environ.get("HOME")

sys.path = [rootPath_to_JaxTutos_localrepository + "JaxTutos", '',               
            rootPath_to_JaxTutos_conda_env + "JaxTutos/lib/python38.zip", 
            rootPath_to_JaxTutos_conda_env + "JaxTutos/lib/python3.8", 
            rootPath_to_JaxTutos_conda_env + "JaxTutos/lib/python3.8/lib-dynload", 
            rootPath_to_JaxTutos_conda_env + "JaxTutos/lib/python3.8/site-packages", 
            homePath+ "/.local/lib/python3.8/site-packages",
            '/opt/conda/lib/python3.8/site-packages'
           ]


