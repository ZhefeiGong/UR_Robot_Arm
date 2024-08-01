#!/usr/bin/env python

import motion_commander
import inference
from motion_commander import MotionCommander
from vla_client import VLAClient



def run():
    
    motion_client = MotionCommander()
    vla_client = VLAClient(host="10.0.2.11", port=30466)

    # IMAGE DEAL


    # 



if __name__ == "__main__":
    run()


