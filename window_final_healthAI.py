import cv2
import numpy as np
import time
import threading
import random
import sys
import os
import shutil
import torch
import torch.nn as nn
import time
from serial.tools import list_ports
from scipy.signal import butter, filtfilt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def my_ECG(i):
    x = '''
[b'7fA0643']
0  -  1
[b'a0A0666']
0  -  1
[b'a6A0481']
0  -  1
[b'wA0655']
0  -  1
[b'a3A0455']
0  -  1
[b'qA0643']
0  -  1
[b'a0A0616']
0  -  1
[b'99A0564']
0  -  1
[b'8cA0655']
0  -  1
[b'a3A0426']
0  -  1
[b'jA0650']
0  -  1
[b'a2A0514']
0  -  1
[b'80A0632']
0  -  1
[b'9dA0654']
0  -  1
[b'a3A0457']
0  -  1
[b'qA0628']
0  -  1
[b'9cA0430']
0  -  1
[b'kA0614']
0  -  1
[b'99A0612']
0  -  1
[b'98A0549']
0  -  1
[b'88A0655']
0  -  1
[b'a3A0425']
0  -  1
[b'iA0663']
0  -  1
[b'a5A0531']
0  -  1
[b'84A0644']
0  -  1
[b'a0A0671']
0  -  1
[b'a7A0460']
0  -  1
[b'rA0658']
0  -  1
[b'a4A0457']
0  -  1
[b'qA0642']
0  -  1
[b'a0A0665']
0  -  1
[b'a5A0556']
0  -  1
[b'8aA0665']
0  -  1
[b'a5A0439']
0  -  1
[b'mA0671']
0  -  1
[b'a7A0552']
0  -  1
[b'89A0661']
0  -  1
[b'a4A0698']
0  -  1
[b'adA0463']
0  -  1
[b'sA0683']
0  -  1
[b'aaA0482']
0  -  1
[b'xA0660']
0  -  1
[b'a4A0663']
0  -  1
[b'a5A0520']
0  -  1
[b'81A0615']
0  -  1
[b'99A0375']
0  -  1
[b']A0584']
0  -  1
[b'91A0469']
0  -  1
[b'tA0552']
0  -  1
[b'89A0589']
0  -  1
[b'92A0353']
0  -  1
[b'WA0590']
0  -  1
[b'93A0409']
0  -  1
[b'eA0598']
0  -  1
[b'95A0627']
0  -  1
[b'9cA0498']
0  -  1
[b'|A0614']
0  -  1
[b'99A0381']
0  -  1
[b'^A0608']
0  -  1
[b'97A0503']
0  -  1
[b'}A0590']
0  -  1
[b'93A0636']
0  -  1
[b'9eA0399']
0  -  1
[b'cA0643']
0  -  1
[b'a0A0462']
0  -  1
[b'sA0647']
0  -  1
[b'a1A0676']
0  -  1
[b'a8A0546']
0  -  1
[b'88A0674']
0  -  1
[b'a8A0454']
0  -  1
[b'qA0677']
0  -  1
[b'a8A0584']
0  -  1
[b'91A0659']
0  -  1
[b'a4A0710']
0  -  1
[b'b0A0466']
0  -  1
[b'tA0714']
0  -  1
[b'b1A0532']
0  -  1
[b'84A0716']
0  -  1
[b'b2A0745']
0  -  1
[b'b9A0600']
0  -  1
[b'95A0735']
0  -  1
[b'b7A0502']
0  -  1
[b'}A0727']
0  -  1
[b'b5A0631']
0  -  1
[b'9dA0690']
0  -  1
[b'abA0742']
0  -  1
[b'b8A0480']
0  -  1
[b'wA0727']
0  -  1
[b'b5A0535']
0  -  1
[b'85A0711']
0  -  1
[b'b1A0734']
0  -  1
[b'b6A0578']
0  -  1
[b'90A0717']
0  -  1
[b'b2A0482']
0  -  1
[b'xA0705']
0  -  1
[b'afA0612']
0  -  1
[b'98A0657']
0  -  1
[b'a3A0715']
0  -  1
[b'b2A0455']
0  -  1
[b'qA0703']
0  -  1
[b'afA0519']
0  -  1
[b'81A0692']
0  -  1
[b'acA0718']
0  -  1
[b'b2A0556']
0  -  1
[b'8aA0702']
0  -  1
[b'aeA0470']
0  -  1
[b'uA0693']
0  -  1
[b'acA0607']
0  -  1
[b'97A0639']
0  -  1
[b'9fA0705']
0  -  1
[b'afA0447']
0  -  1
[b'oA0694']
0  -  1
[b'acA0512']
0  -  1
[b'7fA0685']
0  -  1
[b'aaA0706']
0  -  1
[b'afA0542']
0  -  1
[b'87A0691']
0  -  1
[b'acA0464']
0  -  1
[b'sA0682']
0  -  1
[b'aaA0607']
0  -  1
[b'97A0624']
0  -  1
[b'9bA0696']
0  -  1
[b'adA0436']
0  -  1
[b'lA0684']
0  -  1
[b'aaA0510']
0  -  1
[b'7fA0677']
0  -  1
[b'a8A0699']
0  -  1
[b'aeA0529']
0  -  1
[b'83A0688']
0  -  1
[b'abA0458']
0  -  1
[b'rA0676']
0  -  1
[b'a8A0608']
0  -  1
[b'97A0611']
0  -  1
[b'98A0687']
0  -  1
[b'abA0430']
0  -  1
[b'kA0676']
0  -  1
[b'a8A0504']
0  -  1
[b'}A0665']
0  -  1
[b'a5A0690']
0  -  1
[b'abA0511']
0  -  1
[b'7fA0675']
0  -  1
[b'a8A0451']
0  -  1
[b'pA0665']
0  -  1
[b'a5A0606']
0  -  1
[b'97A0597']
0  -  1
[b'94A0678']
0  -  1
[b'a9A0422']
0  -  1
[b'iA0668']
0  -  1
[b'a6A0504']
0  -  1
[b'}A0660']
0  -  1
[b'a4A0684']
0  -  1
[b'aaA0498']
0  -  1
[b'|A0669']
0  -  1
[b'a6A0447']
0  -  1
[b'oA0659']
0  -  1
[b'a4A0606']
0  -  1
[b'97A0583']
0  -  1
[b'91A0668']
0  -  1
[b'a6A0415']
0  -  1
[b'gA0655']
0  -  1
[b'a3A0501']
0  -  1
[b'|A0644']
0  -  1
[b'a0A0667']
0  -  1
[b'a6A0475']
0  -  1
[b'vA0650']
0  -  1
[b'a2A0431']
0  -  1
[b'kA0639']
0  -  1
[b'9fA0598']
0  -  1
[b'95A0557']
0  -  1
[b'8aA0650']
0  -  1
[b'a2A0400']
0  -  1
[b'cA0639']
0  -  1
[b'9fA0492']
0  -  1
[b'zA0631']
0  -  1
[b'9dA0655']
0  -  1
[b'a3A0462']
0  -  1
[b'sA0643']
0  -  1
[b'a0A0433']
0  -  1
[b'kA0637']
0  -  1
[b'9eA0607']
0  -  1
[b'97A0553']
0  -  1
[b'89A0648']
0  -  1
[b'a1A0400']
0  -  1
[b'cA0638']
0  -  1
[b'9fA0498']
0  -  1
[b'|A0626']
0  -  1
[b'9cA0649']
0  -  1
[b'a1A0451']
0  -  1
[b'pA0635']
0  -  1
[b'9eA0427']
0  -  1
[b'jA0626']
0  -  1
[b'9cA0611']
0  -  1
[b'98A0538']
0  -  1
[b'86A0640']
0  -  1
[b'9fA0397']
0  -  1
[b'bA0631']
0  -  1
[b'9dA0500']
0  -  1
[b'|A0621']
0  -  1
[b'9aA0651']
0  -  1
[b'a2A0445']
0  -  1
[b'nA0643']
0  -  1
[b'a0A0439']
0  -  1
[b'mA0638']
0  -  1
[b'9fA0659']
0  -  1
[b'a4A0543']
0  -  1
[b'87A0650']
0  -  1
[b'a2A0409']
0  -  1
[b'eA0640']
0  -  1
[b'9fA0515']
0  -  1
[b'80A0625']
0  -  1
[b'9bA0656']
0  -  1
[b'a3A0424']
0  -  1
[b'iA0641']
0  -  1
[b'9fA0441']
0  -  1
[b'mA0635']
0  -  1
[b'9eA0657']
0  -  1
[b'a3A0530']
0  -  1
[b'84A0643']
0  -  1
[b'a0A0397']
0  -  1
[b'bA0629']
0  -  1
[b'9cA0510']
0  -  1
[b'7fA0610']
0  -  1
[b'98A0644']
0  -  1
[b'a0A0392']
0  -  1
[b'aA0635']
0  -  1
[b'9eA0438']
0  -  1
[b'mA0631']
0  -  1
[b'9dA0658']
0  -  1
[b'a4A0525']
0  -  1
[b'82A0646']
0  -  1
[b'a1A0405']
0  -  1
[b'dA0636']
0  -  1
[b'9eA0526']
0  -  1
[b'83A0614']
0  -  1
[b'99A0653']
0  -  1
[b'a2A0400']
0  -  1
[b'cA0648']
0  -  1
[b'a1A0451']
0  -  1
[b'pA0642']
0  -  1
[b'a0A0669']
0  -  1
[b'a6A0529']
0  -  1
[b'83A0656']
0  -  1
[b'a3A0419']
0  -  1
[b'hA0648']
0  -  1
[b'a1A0542']
0  -  1
[b'87A0617']
0  -  1
[b'99A0660']
0  -  1
[b'a4A0403']
0  -  1
[b'dA0649']
0  -  1
[b'a1A0459']
0  -  1
[b'rA0641']
0  -  1
[b'9fA0668']
0  -  1
[b'a6A0520']
0  -  1
[b'81A0653']
0  -  1
[b'a2A0416']
0  -  1
[b'gA0644']
0  -  1
[b'a0A0546']
0  -  1
[b'88A0609']
0  -  1
[b'97A0658']
0  -  1
[b'a4A0402']
0  -  1
[b'dA0645']
0  -  1
[b'a0A0459']
0  -  1
[b'rA0640']
0  -  1
[b'9fA0664']
0  -  1
[b'a5A0515']
0  -  1
[b'80A0650']
0  -  1
[b'a2A0420']
0  -  1
[b'hA0645']
0  -  1
[b'a0A0555']
0  -  1
[b'8aA0607']
0  -  1
[b'97A0664']
0  -  1
[b'a5A0406']
0  -  1
[b'eA0654']
0  -  1
[b'a3A0472']
0  -  1
[b'uA0652']
0  -  1
[b'a2A0677']
0  -  1
[b'a8A0518']
0  -  1
[b'81A0663']
0  -  1
[b'a5A0434']
0  -  1
[b'lA0659']
0  -  1
[b'a4A0574']
0  -  1
[b'8fA0614']
0  -  1
[b'99A0678']
0  -  1
[b'a9A0419']
0  -  1
[b'hA0668']
0  -  1
[b'a6A0487']
0  -  1
[b'yA0665']
0  -  1
[b'a5A0689']
0  -  1
[b'abA0524']
0  -  1
[b'82A0677']
0  -  1
[b'a8A0444']
0  -  1
[b'nA0669']
0  -  1
[b'a6A0591']
0  -  1
[b'93A0615']
0  -  1
[b'99A0683']
0  -  1
[b'aaA0421']
0  -  1
[b'hA0673']
0  -  1
[b'a7A0493']
0  -  1
[b'zA0666']
0  -  1
[b'a6A0692']
0  -  1
[b'acA0519']
0  -  1
[b'81A0679']
0  -  1
[b'a9A0441']
0  -  1
[b'mA0670']
0  -  1
[b'a7A0601']
0  -  1
[b'95A0610']
0  -  1
[b'98A0684']
0  -  1
[b'aaA0421']
0  -  1
[b'hA0675']
0  -  1
[b'a8A0495']
0  -  1
[b'{A0666']
0  -  1
[b'a6A0689']
0  -  1
[b'abA0506']
0  -  1
[b'~A0674']
0  -  1
[b'a8A0441']
0  -  1
[b'mA0664']
0  -  1
[b'a5A0602']
0  -  1
[b'96A0597']
0  -  1
[b'94A0676']
0  -  1
[b'a8A0414']
0  -  1
[b'gA0660']
0  -  1
[b'a4A0495']
0  -  1
[b'{A0656']
0  -  1
[b'a3A0679']
0  -  1
[b'a9A0492']
0  -  1
[b'zA0665']
0  -  1
[b'a5A0438']
0  -  1
[b'mA0654']
0  -  1
[b'a3A0604']
0  -  1
[b'96A0582']
0  -  1
[b'91A0667']
0  -  1
[b'a6A0410']
0  -  1
[b'fA0661']
0  -  1
[b'a4A0498']
0  -  1
[b'|A0653']
0  -  1
[b'a2A0676']
0  -  1
[b'a8A0482']
0  -  1
[b'xA0663']
0  -  1
[b'a5A0441']
0  -  1
[b'mA0655']
0  -  1
[b'a3A0614']
0  -  1
[b'99A0579']
0  -  1
[b'90A0647']
0  -  1
[b'a1A0393']
0  -  1
[b'aA0595']
0  -  1
[b'94A0463']
0  -  1
[b'sA0553']
0  -  1
[b'89A0607']
0  -  1
[b'97A0426']
0  -  1
[b'jA0549']
0  -  1
[b'88A0368']
0  -  1
[b'[A0507']
0  -  1
[b'~A0525']
0  -  1
[b'82A0455']
0  -  1
[b'qA0559']
0  -  1
[b'8bA0322']
0  -  1
[b'PA0515']
0  -  1
[b'80A0414']
0  -  1
[b'gA0502']
0  -  1
[b'}A0565']
0  -  1
[b'8cA0377']
0  -  1
[b']A0514']
0  -  1
[b'80A0335']
0  -  1
[b'SA0471']
0  -  1
[b'uA0503']
0  -  1
[b'}A0409']
0  -  1
[b'eA0514']
0  -  1
[b'80A0287']
0  -  1
[b'GA0442']
0  -  1
[b'nA0374']
0  -  1
[b']A0449']
0  -  1
[b'oA0509']
0  -  1
[b'~A0320']
0  -  1
[b'OA0421']
0  -  1
[b'hA0290']
0  -  1
[b'HA0425']
0  -  1
[b'iA0472']
0  -  1
[b'uA0368']
0  -  1
[b'[A0479']
0  -  1
[b'wA0255']
0  -  1
[b'?A0444']
0  -  1
[b'nA0357']
0  -  1
[b'XA0459']
0  -  1
[b'rA0494']
0  -  1
[b'{A0277']
0  -  1
[b'EA0411']
0  -  1
[b'fA0284']
0  -  1
[b'FA0420']
0  -  1
[b'hA0493']
0  -  1
[b'zA0355']
0  -  1
[b'XA0479']
0  -  1
[b'wA0249']
0  -  1
[b'>A0407']
0  -  1
[b'eA0353']
0  -  1
[b'WA0435']
0  -  1
[b'lA0484']
0  -  1
[b'xA0239']
0  -  1
[b';A0393']
0  -  1
[b'aA0269']
0  -  1
[b'CA0404']
0  -  1
[b'dA0488']
0  -  1
[b'yA0334']
0  -  1
[b'SA0440']
0  -  1
[b'mA0226']
0  -  1
[b'8A0377']
0  -  1
[b']A0329']
0  -  1
[b'RA0403']
0  -  1
[b'dA0452']
0  -  1
[b'pA0210']
0  -  1
[b'4A0367']
0  -  1
[b'[A0250']
0  -  1
[b'>A0373']
0  -  1
[b'\\A0458']
0  -  1
[b'rA0309']
0  -  1
[b'MA0417']
0  -  1
[b'gA0206']
0  -  1
[b'3A0367']
0  -  1
[b'[A0323']
0  -  1
[b'PA0355']
0  -  1
[b'XA0434']
0  -  1
[b'lA0191']
0  -  1
[b'/A0350']
0  -  1
[b'WA0236']
0  -  1
[b':A0362']
0  -  1
[b'ZA0437']
0  -  1
[b'lA0365']
0  -  1
[b'ZA0440']
0  -  1
[b'mA0315']
0  -  1
[b'NA0443']
0  -  1
[b'nA0389']
0  -  1
[b'`A0428']
0  -  1
[b'jA0464']
0  -  1
[b'sA0317']
0  -  1
[b'OA0452']
0  -  1
[b'pA0354']
0  -  1
[b'XA0443']
0  -  1
[b'nA0469']
0  -  1
[b'tA0394']
0  -  1
[b'bA0459']
0  -  1
[b'rA0333']
0  -  1
[b'SA0443']
0  -  1
[b'nA0391']
0  -  1
[b'aA0419']
0  -  1
[b'hA0455']
0  -  1
[b'qA0318']
0  -  1
[b'OA0446']
0  -  1
[b'oA0355']
0  -  1
[b'XA0435']
0  -  1
[b'lA0458']
0  -  1
[b'rA0386']
0  -  1
[b'`A0446']
0  -  1
[b'oA0334']
0  -  1
[b'SA0434']
0  -  1
[b'lA0390']
0  -  1
[b'aA0411']
0  -  1
[b'fA0445']
0  -  1
[b'nA0320']
0  -  1
[b'OA0434']
0  -  1
[b'lA0352']
0  -  1
[b'WA0414']
0  -  1
[b'gA0429']
0  -  1
[b'jA0367']
0  -  1
[b'[A0426']
0  -  1
[b'jA0332']
0  -  1
[b'RA0416']
0  -  1
[b'gA0383']
0  -  1
[b'_A0404']
0  -  1
[b'dA0438']
0  -  1
[b'mA0338']
0  -  1
[b'TA0434']
0  -  1
[b'lA0374']
0  -  1
[b']A0425']
0  -  1
[b'iA0451']
0  -  1
[b'pA0405']
0  -  1
[b'dA0445']
0  -  1
[b'nA0375']
0  -  1
[b']A0433']
0  -  1
[b'kA0408']
0  -  1
[b'eA0423']
0  -  1
[b'iA0448']
0  -  1
[b'oA0378']
0  -  1
[b'^A0440']
0  -  1
[b'mA0402']
0  -  1
[b'dA0416']
0  -  1
[b'gA0420']
0  -  1
[b'hA0376']
0  -  1
[b']A0347']
0  -  1
[b'VA0256']
0  -  1
[b'?A0219']
0  -  1
[b'6A0224']
0  -  1
[b'7A0294']
0  -  1
[b'IA0388']
0  -  1
[b'`A0393']
0  -  1
[b'aA0421']
0  -  1
[b'hA0410']
0  -  1
[b'fA0407']
0  -  1
[b'eA0423']
0  -  1
[b'iA0430']
0  -  1
[b'kA0435']
0  -  1
[b'lA0433']
0  -  1
[b'kA0458']
0  -  1
[b'rA0461']
0  -  1
[b'rA0491']
0  -  1
[b'zA0511']
0  -  1
[b'7fA0482']
0  -  1
[b'xA0510']
0  -  1
[b'7fA0492']
0  -  1
[b'zA0503']
0  -  1
[b'}A0529']
0  -  1
[b'83A0523']
0  -  1
[b'82A0529']
0  -  1
[b'83A0500']
0  -  1
[b'|A0520']
0  -  1
[b'81A0505']
0  -  1
[b'}A0523']
0  -  1
[b'82A0537']
0  -  1
[b'85A0497']
0  -  1
[b'{A0528']
0  -  1
[b'83A0504']
0  -  1
[b'}A0513']
0  -  1
[b'7fA0533']
0  -  1
[b'84A0523']
0  -  1
[b'82A0521']
0  -  1
[b'81A0496']
0  -  1
[b'{A0501']
0  -  1
[b'|A0481']
0  -  1
[b'wA0496']
0  -  1
[b'{A0491']
0  -  1
[b'zA0468']
0  -  1
[b'tA0468']
0  -  1
[b'tA0456']
0  -  1
[b'qA0459']
0  -  1
[b'rA0493']
0  -  1
[b'zA0520']
0  -  1
[b'81A0512']
0  -  1
[b'7fA0526']
0  -  1
[b'83A0518']
0  -  1
[b'81A0518']
0  -  1
[b'81A0558']
0  -  1
[b'8bA0557']
0  -  1
[b'8aA0579']
0  -  1
[b'90A0563']
0  -  1
[b'8cA0582']
0  -  1
[b'91A0584']
0  -  1
[b'91A0622']
0  -  1
[b'9bA0670']
0  -  1
[b'a7A0630']
0  -  1
[b'9dA0664']
0  -  1
[b'a5A0620']
0  -  1
[b'9aA0643']
0  -  1
[b'a0A0666']
0  -  1
[b'a6A0637']
0  -  1
[b'9eA0679']
0  -  1
[b'a9A0625']
0  -  1
[b'9bA0637']
0  -  1
[b'9eA0606']
0  -  1
[b'97A0627']
0  -  1
[b'9cA0664']
0  -  1
[b'a5A0617']
0  -  1
[b'99A0645']
0  -  1
[b'a0A0598']
0  -  1
[b'95A0617']
0  -  1
[b'99A0633']
0  -  1
[b'9dA0608']
0  -  1
[b'97A0630']
0  -  1
[b'9dA0593']
0  -  1
[b'93A0595']
0  -  1
[b'94A0577']
0  -  1
[b'8fA0597']
0  -  1
[b'94A0599']
0  -  1
[b'95A0586']
0  -  1
[b'92A0601']
0  -  1
[b'95A0567']
0  -  1
[b'8dA0585']
0  -  1
[b'91A0596']
0  -  1
[b'94A0577']
0  -  1
[b'8fA0580']
0  -  1
[b'90A0563']
0  -  1
[b'8cA0554']
0  -  1
[b'8aA0553']
0  -  1
[b'89A0571']
0  -  1
[b'8eA0554']
0  -  1
[b'8aA0566']
0  -  1
[b'8dA0561']
0  -  1
[b'8bA0556']
0  -  1
[b'8aA0577']
0  -  1
[b'8fA0581']
0  -  1
[b'90A0578']
0  -  1
[b'90A0558']
0  -  1
[b'8bA0572']
0  -  1
[b'8eA0550']
0  -  1
[b'89A0562']
0  -  1
[b'8cA0582']
0  -  1
[b'91A0533']
0  -  1
[b'84A0576']
0  -  1
[b'8fA0544']
0  -  1
[b'87A0567']
0  -  1
[b'8dA0588']
0  -  1
[b'92A0570']
0  -  1
[b'8eA0584']
0  -  1
[b'91A0529']
0  -  1
[b'83A0577']
0  -  1
[b'8fA0544']
0  -  1
[b'87A0569']
0  -  1
[b'8dA0594']
0  -  1
[b'94A0470']
0  -  1
[b'uA0526']
0  -  1
[b'83A0415']
0  -  1
[b'gA0504']
0  -  1
[b'}A0495']
0  -  1
[b'{A0405']
0  -  1
[b'dA0513']
0  -  1
[b'7fA0335']
0  -  1
[b'SA0462']
0  -  1
[b'sA0420']
0  -  1
[b'hA0443']
0  -  1
[b'nA0515']
0  -  1
[b'80A0314']
0  -  1
[b'NA0431']
0  -  1
[b'kA0357']
0  -  1
[b'XA0445']
0  -  1
[b'nA0477']
0  -  1
[b'vA0400']
0  -  1
[b'cA0498']
0  -  1
[b'|A0395']
0  -  1
[b'bA0461']
0  -  1
[b'rA0446']
0  -  1
[b'oA0465']
0  -  1
[b'sA0549']
0  -  1
[b'88A0356']
0  -  1
[b'XA0459']
0  -  1
[b'rA0383']
0  -  1
[b'_A0469']
0  -  1
[b'tA0467']
0  -  1
[b'tA0411']
0  -  1
[b'fA0484']
0  -  1
[b'xA0393']
0  -  1
[b'aA0458']
0  -  1
[b'rA0446']
0  -  1
[b'oA0460']
0  -  1
[b'rA0529']
0  -  1
[b'83A0407']
0  -  1
[b'eA0464']
0  -  1
[b'sA0425']
0  -  1
[b'iA0485']
0  -  1
[b'xA0508']
0  -  1
[b'~A0430']
0  -  1
[b'kA0513']
0  -  1
[b'7fA0413']
0  -  1
[b'fA0474']
0  -  1
[b'vA0450']
0  -  1
[b'pA0463']
0  -  1
[b'sA0530']
0  -  1
[b'84A0410']
0  -  1
[b'fA0473']
0  -  1
[b'uA0473']
0  -  1
[b'uA0529']
0  -  1
[b'83A0552']
0  -  1
[b'89A0511']
0  -  1
[b'7fA0547']
0  -  1
[b'88A0475']
0  -  1
[b'vA0539']
0  -  1
[b'86A0509']
0  -  1
[b'~A0525']
0  -  1
[b'82A0552']
0  -  1
[b'89A0464']
0  -  1
[b'sA0544']
0  -  1
[b'87A0488']
0  -  1
[b'yA0532']
0  -  1
[b'84A0554']
0  -  1
[b'8aA0508']
0  -  1
[b'~A0549']
0  -  1
[b'88A0472']
0  -  1
[b'uA0539']
0  -  1
[b'86A0511']
0  -  1
[b'7fA0450']
0  -  1
[b'pA0564']
0  -  1
[b'8cA0332']
0  -  1
[b'RA0467']
0  -  1
[b'tA0367']
0  -  1
[b'[A0447']
0  -  1
[b'oA0516']
0  -  1
[b'80A0372']
0  -  1
[b'\\A0413']
0  -  1
[b'fA0282']
0  -  1
[b'FA0403']
0  -  1
[b'dA0400']
0  -  1
[b'cA0354']
0  -  1
[b'XA0458']
0  -  1
[b'rA0284']
0  -  1
[b'FA0379']
0  -  1
[b'^A0332']
0  -  1
[b'RA0412']
0  -  1
[b'fA0407']
0  -  1
[b'eA0374']
0  -  1
[b']A0426']
0  -  1
[b'jA0360']
0  -  1
[b'YA0422']
0  -  1
[b'iA0450']
0  -  1
[b'pA0418']
0  -  1
[b'hA0493']
0  -  1
[b'zA0375']
0  -  1
[b']A0430']
0  -  1
[b'kA0382']
0  -  1
[b'_A0438']
0  -  1
[b'mA0462']
0  -  1
[b'sA0398']
0  -  1
[b'cA0468']
0  -  1
[b'tA0395']
0  -  1
[b'bA0449']
0  -  1
[b'oA0467']
0  -  1
[b'tA0427']
0  -  1
[b'jA0509']
0  -  1
[b'~A0400']
0  -  1
[b'cA0450']
0  -  1
[b'pA0423']
0  -  1
[b'iA0465']
0  -  1
[b'sA0475']
0  -  1
[b'vA0417']
0  -  1
[b'gA0460']
0  -  1
[b'rA0405']
0  -  1
[b'dA0458']
0  -  1
[b'rA0448']
0  -  1
[b'oA0420']
0  -  1
[b'hA0502']
0  -  1
[b'}A0390']
0  -  1
[b'aA0447']
0  -  1
[b'oA0432']
0  -  1
[b'kA0469']
0  -  1
[b'tA0525']
0  -  1
[b'82A0419']
0  -  1
[b'hA0567']
0  -  1
[b'8dA0442']
0  -  1
[b'nA0538']
0  -  1
[b'86A0563']
0  -  1
[b'8cA0460']
0  -  1
[b'rA0570']
0  -  1
[b'8eA0367']
0  -  1
[b'[A0555']
0  -  1
[b'8aA0478']
0  -  1
[b'wA0515']
0  -  1
[b'80A0575']
0  -  1
[b'8fA0347']
0  -  1
[b'VA0505']
0  -  1
[b'}A0404']
0  -  1
[b'dA0545']
0  -  1
[b'87A0575']
0  -  1
[b'8fA0437']
0  -  1
[b'lA0552']
0  -  1
[b'89A0360']
0  -  1
[b'YA0538']
0  -  1
[b'86A0477']
0  -  1
[b'vA0491']
0  -  1
[b'zA0533']
0  -  1
[b'84A0316']
0  -  1
[b'NA0469']
0  -  1
[b'tA0381']
0  -  1
[b'^A0506']
0  -  1
[b'~A0549']
0  -  1
[b'88A0404']
0  -  1
[b'dA0525']
0  -  1
[b'82A0334']
0  -  1
[b'SA0474']
0  -  1
[b'vA0455']
0  -  1
[b'qA0475']
0  -  1
[b'vA0530']
0  -  1
[b'84A0306']
0  -  1
[b'LA0461']
0  -  1
[b'rA0370']
0  -  1
[b'\\A0496']
0  -  1
[b'{A0503']
0  -  1
[b'}A0390']
0  -  1
[b'aA0507']
0  -  1
[b'~A0327']
0  -  1
[b'QA0476']
0  -  1
[b'vA0451']
0  -  1
[b'pA0453']
0  -  1
[b'pA0520']
0  -  1
[b'81A0303']
0  -  1
[b'KA0470']
0  -  1
[b'uA0377']
0  -  1
[b']A0499']
0  -  1
[b'|A0527']
0  -  1
[b'83A0380']
0  -  1
[b'^A0490']
0  -  1
[b'zA0318']
0  -  1
[b'OA0487']
0  -  1
[b'yA0451']
0  -  1
[b'pA0438']
0  -  1
[b'mA0516']
0  -  1
[b'80A0296']
0  -  1
[b'IA0465']
0  -  1
[b'sA0378']
0  -  1
[b'^A0486']
0  -  1
[b'yA0528']
0  -  1
[b'83A0376']
0  -  1
[b']A0496']
0  -  1
[b'{A0336']
0  -  1
[b'SA0485']
0  -  1
[b'xA0458']
0  -  1
[b'rA0431']
0  -  1
[b'kA0512']
0  -  1
[b'7fA0304']
0  -  1
[b'KA0463']
0  -  1
[b'sA0395']
0  -  1
[b'bA0467']
0  -  1
[b'tA0524']
0  -  1
[b'82A0370']
0  -  1
[b'\\A0497']
0  -  1
[b'{A0341']
0  -  1
[b'UA0485']
0  -  1
[b'xA0463']
0  -  1
[b'sA0428']
0  -  1
[b'jA0519']
0  -  1
[b'81A0305']
0  -  1
[b'LA0503']
0  -  1
[b'}A0402']
0  -  1
[b'dA0497']
0  -  1
[b'{A0533']
0  -  1
[b'84A0372']
0  -  1
[b'\\A0508']
0  -  1
[b'~A0344']
0  -  1
[b'UA0486']
0  -  1
[b'yA0472']
0  -  1
[b'uA0433']
0  -  1
[b'kA0532']
0  -  1
[b'84A0317']
0  -  1
[b'OA0474']
0  -  1
[b'vA0414']
0  -  1
[b'gA0486']
0  -  1
[b'yA0540']
0  -  1
[b'86A0375']
0  -  1
[b']A0501']
0  -  1
[b'|A0349']
0  -  1
[b'VA0500']
0  -  1
[b'|A0483']
0  -  1
[b'xA0434']
0  -  1
[b'lA0527']
0  -  1
[b'83A0315']
0  -  1
[b'NA0468']
0  -  1
[b'tA0412']
0  -  1
[b'fA0474']
0  -  1
[b'vA0534']
0  -  1
[b'85A0363']
0  -  1
[b'ZA0503']
0  -  1
[b'}A0338']
0  -  1
[b'TA0514']
0  -  1
[b'80A0482']
0  -  1
[b'xA0431']
0  -  1
[b'kA0525']
0  -  1
[b'82A0301']
0  -  1
[b'KA0480']
0  -  1
[b'wA0384']
0  -  1
[b'_A0485']
0  -  1
[b'xA0538']
0  -  1
[b'86A0359']
0  -  1
[b'YA0535']
0  -  1
[b'85A0337']
0  -  1
[b'TA0531']
0  -  1
[b'84A0500']
0  -  1
[b'|A0452']
0  -  1
[b'pA0549']
0  -  1
[b'88A0313']
0  -  1
[b'NA0540']
0  -  1
[b'86A0412']
0  -  1
[b'fA0500']
0  -  1
[b'|A0563']
0  -  1
[b'8cA0375']
0  -  1
[b']A0536']
0  -  1
[b'85A0350']
0  -  1
[b'WA0514']
0  -  1
[b'80A0518']
0  -  1
[b'81A0453']
0  -  1
[b'pA0560']
0  -  1
[b'8bA0329']
0  -  1
[b'RA0526']
0  -  1
[b'83A0431']
0  -  1
[b'kA0522']
0  -  1
[b'82A0563']
0  -  1
[b'8cA0359']
0  -  1
[b'YA0529']
0  -  1
[b'83A0365']
0  -  1
[b'ZA0502']
0  -  1
[b'}A0519']
0  -  1
[b'81A0444']
0  -  1
[b'nA0557']
0  -  1
[b'8aA0333']
0  -  1
[b'SA0537']
0  -  1
[b'85A0436']
0  -  1
[b'lA0520']
0  -  1
[b'81A0568']
0  -  1
[b'8dA0345']
0  -  1
[b'UA0556']
0  -  1
[b'8aA0358']
0  -  1
[b'YA0470']
0  -  1
[b'uA0483']
0  -  1
[b'xA0390']
0  -  1
[b'aA0499']
0  -  1
[b'|A0291']
0  -  1
[b'HA0428']
0  -  1
[b'jA0352']
0  -  1
[b'WA0416']
0  -  1
[b'gA0487']
0  -  1
[b'yA0294']
0  -  1
[b'IA0421']
0  -  1
[b'hA0272']
0  -  1
[b'CA0389']
0  -  1
[b'`A0404']
0  -  1
[b'dA0319']
0  -  1
[b'OA0435']
0  -  1
[b'lA0226']
0  -  1
[b'8A0365']
0  -  1
[b'ZA0305']
0  -  1
[b'LA0384']
0  -  1
[b'_A0391']
0  -  1
[b'aA0312']
0  -  1
[b'MA0391']
0  -  1
[b'aA0310']
0  -  1
[b'MA0391']
0  -  1
[b'aA0436']
0  -  1
[b'lA0343']
0  -  1
[b'UA0460']
0  -  1
[b'rA0293']
0  -  1
[b'IA0389']
0  -  1
[b'`A0343']
0  -  1
[b'UA0454']
0  -  1
[b'qA0494']
0  -  1
[b'{A0263']
0  -  1
[b'AA0424']
0  -  1
[b'iA0347']
0  -  1
[b'VA0453']
0  -  1
[b'pA0501']
0  -  1
[b'|A0359']
0  -  1
[b'YA0482']
0  -  1
[b'xA0308']
0  -  1
[b'LA0465']
0  -  1
[b'sA0425']
0  -  1
[b'iA0417']
0  -  1
[b'gA0493']
0  -  1
[b'zA0282']
0  -  1
[b'FA0428']
0  -  1
[b'jA0365']
0  -  1
[b'ZA0436']
0  -  1
[b'lA0504']
0  -  1
[b'}A0362']
0  -  1
[b'ZA0475']
0  -  1
[b'vA0313']
0  -  1
[b'NA0464']
0  -  1
[b'sA0425']
0  -  1
[b'iA0415']
0  -  1
[b'gA0492']
0  -  1
[b'zA0284']
0  -  1
[b'FA0425']
0  -  1
[b'iA0366']
0  -  1
[b'[A0459']
0  -  1
[b'rA0503']
0  -  1
[b'}A0360']
0  -  1
[b'YA0474']
0  -  1
[b'vA0314']
0  -  1
[b'NA0465']
0  -  1
[b'sA0430']
0  -  1
[b'kA0415']
0  -  1
[b'gA0490']
0  -  1
[b'zA0289']
0  -  1
[b'HA0444']
0  -  1
[b'nA0374']
0  -  1
[b']A0429']
0  -  1
[b'jA0508']
0  -  1
[b'~A0368']
0  -  1
[b'[A0476']
0  -  1
[b'vA0337']
0  -  1
[b'TA0488']
0  -  1
[b'yA0450']
0  -  1
[b'pA0432']
0  -  1
[b'kA0513']
0  -  1
[b'7fA0317']
0  -  1
[b'OA0457']
0  -  1
[b'qA0403']
0  -  1
[b'dA0484']
0  -  1
[b'xA0529']
0  -  1
[b'83A0386']
0  -  1
[b'`A0507']
0  -  1
[b'~A0364']
0  -  1
[b'ZA0498']
0  -  1
[b'|A0474']
0  -  1
[b'vA0449']
0  -  1
[b'oA0541']
0  -  1
[b'86A0348']
0  -  1
[b'VA0485']
0  -  1
[b'xA0432']
0  -  1
[b'kA0503']
0  -  1
[b'}A0561']
0  -  1
[b'8bA0360']
0  -  1
[b'YA0542']
0  -  1
[b'87A0402']
0  -  1
[b'dA0501']
0  -  1
[b'|A0489']
0  -  1
[b'yA0383']
0  -  1
[b'_A0443']
0  -  1
[b'nA0391']
0  -  1
[b'aA0459']
0  -  1
[b'rA0418']
0  -  1
[b'hA0475']
0  -  1
[b'vA0440']
0  -  1
[b'mA0379']
0  -  1
[b'^A0447']
0  -  1
[b'oA0391']
0  -  1
[b'aA0451']
0  -  1
[b'pA0474']
0  -  1
[b'vA0413']
0  -  1
[b'fA0493']
0  -  1
[b'zA0404']
0  -  1
[b'dA0456']
0  -  1
[b'qA0448']
0  -  1
[b'oA0483']
0  -  1
[b'xA0500']
0  -  1
[b'|A0383']
0  -  1
[b'_A0443']
0  -  1
[b'nA0379']
0  -  1
[b'^A0438']
0  -  1
[b'mA0473']
0  -  1
[b'uA0405']
0  -  1
[b'dA0464']
0  -  1
[b'sA0361']
0  -  1
[b'YA0400']
0  -  1
[b'cA0433']
0  -  1
[b'kA0419']
0  -  1
[b'hA0491']
0  -  1
[b'zA0372']
0  -  1
[b'\\A0403']
0  -  1
[b'dA0406']
0  -  1
[b'eA0421']
0  -  1
[b'hA0466']
0  -  1
[b'tA0403']
0  -  1
[b'dA0469']
0  -  1
[b'tA0409']
0  -  1
[b'eA0411']
0  -  1
[b'fA0429']
0  -  1
[b'jA0441']
0  -  1
[b'mA0481']
0  -  1
[b'wA0371']
0  -  1
[b'\\A0413']
0  -  1
[b'fA0394']
0  -  1
[b'bA0428']
0  -  1
[b'jA0449']
0  -  1
[b'oA0392']
0  -  1
[b'aA0393']
0  -  1
[b'aA0365']
0  -  1
[b'ZA0402']
0  -  1
[b'dA0403']
0  -  1
[b'dA0424']
0  -  1
[b'iA0395']
0  -  1
[b'bA0371']
0  -  1
[b'\\A0391']
0  -  1
[b'aA0365']
0  -  1
[b'ZA0405']
0  -  1
[b'dA0424']
0  -  1
[b'iA0376']
0  -  1
[b']A0377']
0  -  1
[b']A0348']
0  -  1
[b'VA0388']
0  -  1
[b'`A0375']
0  -  1
[b']A0406']
0  -  1
[b'eA0362']
0  -  1
[b'ZA0353']
0  -  1
[b'WA0379']
0  -  1
[b'^A0347']
0  -  1
[b'VA0389']
0  -  1
[b'`A0359']
0  -  1
[b'YA0359']
0  -  1
[b'YA0363']
0  -  1
[b'ZA0336']
0  -  1
[b'SA0372']
0  -  1
[b'\\A0377']
0  -  1
    '''
    y = x.split("0  -  1")
    return y[i].split()

class ResBlock(nn.Module):
    def __init__(self, C, kernel=9, dilation=1):
        super().__init__()
        pad = (kernel//2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(C, C, kernel_size=kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(C),
            nn.ReLU(inplace=True),
            nn.Conv1d(C, C, kernel_size=kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(C),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))

class ResNet1D(nn.Module):
    def __init__(self, in_ch=2, num_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.block1 = ResBlock(64, kernel=3, dilation=1)
        self.block2 = ResBlock(64, kernel=3, dilation=2)
        self.block3 = ResBlock(64, kernel=3, dilation=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)













def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(sig, fs, lowcut=0.3, highcut=10.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, sig)



classes = ['AV blocada', 'fibril', 'infarct', 'norm']
model_path = "ml_cardiogram_resnet_3_0.pth"


model = ResNet1D(in_ch=2, num_classes=len(classes)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def predict_one(pulse_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = ['AV blocada', 'fibril', 'infarct', 'norm']
    # Берём последние 2500 отсчётов
    sig = np.array(pulse_data[-2500:])

    # Производная сигнала
    d = np.diff(sig, prepend=sig[0]).astype(np.float32)

    # Создаём 2 канала: [сигнал, производная]
    x = np.stack([sig, d], axis=0)  # (2, 2500)

    # Приводим к формату (1, 2, 2500)
    x = torch.from_numpy(x).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        print(f'answer: {classes}, - {probs}')
        ans = []
        for x, y in zip(classes, probs):
            ans.append([y, x])
        print(ans)
        return ans






shutil.rmtree('card_for_code', ignore_errors=True)

os.makedirs("card_for_code")
print(time.time())
print(time.time())

'''
import create_triangels
import vis
import ecgs
import datchik
import chek
'''
from get_dano_datch import*
from res_net_MLs import*
from datchiki_detect import*
from check_verdict_AI import*


#ECG = [(0.999, "IM"), (0.3000, "AV"), (0.004, "FIB"), (0.004, "NORM")]
vision = [(0.287, "cianoz"), (0.2000, "allergia"), (0.904, "vein"), (0.09, "NORM")]
# check — перепроверка ECG (label, prob)
# datchicky: main detected label + list of related labels + sensor values dict




#sample = {"SpO2":100.0, "MAP":58.0, "HR":80.0, "EtCO2":38.0, "CVP":93.0, "Urine":8.0, "Temp":36.0, "PI":0.7}
#datchicky = evaluate_reading(sample)
#print("Result:", datchicky)


#datchicky = ("Hypoxy", ["shok", "IM", "cianoz"], {"spo2": 85, "BP": 90, "pulse": 110})


def shoot( s1, s2, im, af, av):
    x = check1(im, af, av)
    print(s1)
    print(x)
    if x in [s1[1], s2[1]]:
        if x == s1[1]:
            return [s1[0], s1[1]]
        elif x == s2[1]:
            return [s2[0], s2[1]]
    return [s1[0], s1[1]]




def target(ecg, vis, datch, sample):
    dat = None
    norma_answer = False

    answer = [0, None]
    ecg_s = sorted(ecg)
    print()
    print(ecg)
    if ecg_s[-1][0]>0.35:

        if (ecg_s[-1][0]-ecg_s[-2][0])<=0.08:
            im =[[[i[0] for i in ecg if i[1]=='infarct'][0], sample['MAP'] ,sample["HR"] ,max([i[0] for i in vis if i[1] in ['vein', 'cianoz']])]]
            af =[[[i[0] for i in ecg if i[1]=='fibril'][0], sample['PI'] ,sample["SpO2"] ,max([i[0] for i in vis if i[1] in ['vein', 'cianoz']])]]
            av =[[[i[0] for i in ecg if i[1]=='AV blocada'][0], sample['HR'] ,sample["MAP"] ,max([i[0] for i in vis if i[1] in ['vein', 'cianoz']])]]
            answer = shoot(ecg_s[-1], ecg_s[-2], im, af, av)
        else:
            answer = [ecg_s[-1][0], ecg_s[-1][1]]
        if answer[1] in datch['conflicts'] or answer[1]==datch['diagnosis']:
            print()
            print("datch conf")
            answer[0] = answer[0]*1.1
        else:

            print()
            print("NO datch conf")
            answer[0] = answer[0]/1.2
    if answer[1] in ["Normal", 'norm']:
        answer = [0, None]
    print()
    print(answer)
    print()
    vis_answer = [0, None]
    vis_s = sorted(vis)
    if vis_s[-1][0]>0.45:
        if (vis_s[-1][0]-vis_s[-2][0])<=0.8:
            if (vis_s[-1] in datch['conflicts'] or vis_s[-1] == datch['diagnosis']) and (vis_s[-2] in datch['conflicts'] or vis_s[-2] == datch['diagnosis']):
                vis_answer = [vis_s[-1][0], vis_s[-1][1]]
                if vis_s[-1] == datch[0]:
                    vis_answer[0] = vis_answer[0]*1.5
                else:
                    vis_answer[0] = vis_answer[0] / 2
    if vis_answer[1]=='NORM':
        vis_answer = [0, None]
    print(vis_answer)


    dat_answer = [0, None]
    if datchicky['diagnosis'] not in ["Myocardial infarction", "Third-degree AV block", "Cyanosis (isolated)", "Jugular venous distension (JVD)", 'Second-degree AV block', 'First-degree AV block', "vein", "Atrial fibrillation (proxy)", 'No rule matched', 'norm']:
        dat_answer = [0.5, datchicky['diagnosis']]
        dat = datchicky['explanation']
    if dat_answer[1]=="Normal":
        dat_answer=[0, None]


    if [dat_answer[0], vis_answer[0], answer[0]] in [[0, 0, 0], [0, 'norm', "Normal"], ['No rule matched', 'norm', 0], ['No rule matched', 'norm', 'Normal']] :
        norma_answer = True

    print([dat_answer[0], vis_answer[0], answer[0]])
    print([dat_answer[1], vis_answer[1], answer[1]])





    return answer, vis_answer, dat_answer, norma_answer, dat

'''

# ---------- звук ----------
#try:
import winsound
HAVE_WINSOUND = True
#except ImportError:
 #   HAVE_WINSOUND = False

def beep_forever():
    """Постоянный фоновый звук в отдельном потоке."""
    while True:
        if HAVE_WINSOUND:
            winsound.Beep(800, 200)  # частота 800 Гц, длительность 200 мс
            time.sleep(0.2)
        else:
            sys.stdout.write('\a')
            sys.stdout.flush()
            time.sleep(0.5)
'''
import os

import time
import threading
import os
import platform
import numpy as np
import sounddevice as sd
import threading
import time

# Глобальный флаг для включения/выключения тревоги
alarm_active = False
alarm_thread = None


def play_alarm(frequency=1500, duration=0.3, period=0.8, volume=0.9):
    """
    Воспроизводит медицинский тревожный "beep" пока alarm_active = True

    frequency — частота тона (Гц), для медсигналов обычно 1200–1600
    duration  — длительность самого "пика" (сек)
    period    — период повтора (сек)
    volume    — громкость
    """

    global alarm_active

    fs = 44100  # частота дискретизации

    beep_samples = int(duration * fs)
    t = np.linspace(0, duration, beep_samples, False)

    # Резкий медицинский тон
    waveform = (np.sin(2 * np.pi * frequency * t)).astype(np.float32) * volume

    while alarm_active:
        sd.play(waveform, fs, blocking=True)  # воспроизведение beep
        time.sleep(period - duration)  # интервал между сигналами


def start_alarm():
    """Запускает тревожный сигнал в отдельном потоке."""
    global alarm_active, alarm_thread

    if alarm_active:
        return  # уже играет

    alarm_active = True
    alarm_thread = threading.Thread(target=play_alarm, daemon=True)
    alarm_thread.start()


def stop_alarm():
    """Останавливает тревожный сигнал."""
    global alarm_active
    alarm_active = False
    sd.stop()


'''
print("Включаю тревожный сигнал!")
start_alarm()

time.sleep(10)
'''


# запуск фонового потока со звуком

# ---------- визуал ----------
W, H = 800, 500
cv2.namedWindow("Monitor", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Monitor", W, H)

alphaq = 0.0
dir_alpha = 1
cb1 = 0
times = time.time()
time.sleep(2)
flag = True
from scipy.signal import resample

def resample_ecg(signal, target_len=10000):
    """Ресемплинг ЭКГ до нужной длины"""
    return resample(signal, target_len)
def ecg1(signal):
    print(len(signal))



    plt.figure(figsize=(4, 2))
    plt.plot(signal, label="ECG сигнал", linewidth=1)
    plt.title(f"ECG пример")
    plt.xlabel("Время (отсчёты)")
    plt.ylabel("Амплитуда")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()  # вместо bbox_inches='tight' в savefig

    file_path = f"{'card_for_code'}/plot_{1}.png"
    plt.savefig(file_path)


    '''

    plt.tight_layout()  # вместо bbox_inches='tight' в savefig

    file_path = f"card_for_code/plot_1.png"

    plt.savefig(
        file_path,
        dpi=150,                    # хорошее качество
        bbox_inches='tight',        # можно оставить, но с pad_inches
        pad_inches=0.1,             # важно! даёт отступы вокруг легенды
        facecolor='white',          # белый фон (убирает серость)
        edgecolor='none',           # убирает рамку
        transparent=False           # явно непрозрачный фон
    )
    '''
    plt.close()







def meadly_text(text, coord, color):
    # setup text
    font = cv2.FONT_HERSHEY_SIMPLEX

    # get boundary of this text
    textsize = cv2.getTextSize(text, font, 1, 2)[0]

    # get coords based on boundary
    textX = int((frame.shape[1] - textsize[0]) / 2)

    # add text centered on image
    cv2.putText(frame, text, (textX, coord), font, 1, color, 2)


'''
ports = list(serial.tools.list_ports.comports())

for port in ports:
    print(f"Порт: {port.device}")
    print(f"Описание: {port.description}")
    print(f"Производитель: {port.manufacturer}\n")
    '''
tempF= []

pressure=[]

#arduinoData = serial.Serial('/dev/cu.usbmodem144201', 9600) #Creating our serial object named arduinoData

plt.ion() #Tell matplotlib you want interactive mode to plot live data

cnt=0
number_i = 0

pulse_data = np.array([float(0) for i in range(2508)]).astype(np.float32)

while True:




    '''
    while (arduinoData.inWaiting()==0): #Wait here until there is data

        pass #do nothing

    arduinoString = arduinoData.readline() #read the line of text from the serial port

    dataArray = arduinoString.split() #Split it into an array called dataArray
    print(dataArray)
    pulse_data = np.append(pulse_data, float(dataArray[-3:]))
    '''


    dataArray = [str(random.randint(-2, 4)) for i in range(30)]
    for i in range(25):
        pulse_data = np.append(pulse_data, float(dataArray[i]))
    print(dataArray)

    '''
    for i in range(25):
        try:
            pulse_data = np.append(pulse_data, float(my_ECG(number_i)[0][-5:-2]))
            number_i += 1
        except:
            print(number_i)
            number_i =0

    print(pulse_data[-1])
    #'''


    c=0
    x = []
    '''
    try:
        np.append(pulse_data, float(dataArray[0][3:]))
    except:
        c+=1
        '''



  #  pulse_data = resample_ecg(pulse_data, target_len=500)
    pulse_data = bandpass_filter(pulse_data[-2500:], 450)
    pulse_data.astype(np.float32)
 #   print(pulse_data)
    verdict = predict_one(pulse_data)
  #  print(len(pulse_data))
    ecg1(pulse_data[pulse_data.size-2500:])
    print("1234", verdict)



   # plt.pause(.000001) #Pause Briefly. Important to keep drawnow from crashing

    cnt=cnt+1




    if flag:
    # Тестовая вероятность (потом заменишь своей)\
        if time.time() - times > 1:
            times = time.time()
            ECG = verdict
           # print(ECG)

            sim = SensorSimulator()
            sample = sim.generate()
            datchicky = evaluate_reading(sample)
            print(sample)
            print(datchicky)
            # sample = {"SpO2":100.0, "MAP":58.0, "HR":80.0, "EtCO2":38.0, "CVP":93.0, "Urine":8.0, "Temp":36.0, "PI":0.7}
            # datchicky = evaluate_reading(sample)
            answer_ecg, vis_answer, dat_answer, norma_answer, dat = target(ECG, vision, datchicky, sample)  # <-- вот сюда потом подставишь свою функцию

    frame = np.zeros((H, W, 3), dtype=np.uint8)

    # Цвет и текст по состоянию
    if norma_answer:

        text = "NORM"
        stop_alarm()

        color = (0, 255, 0)  # зелёный
        base_color = (10, 30, 10)
        verdict = []
    elif max([answer_ecg[0], vis_answer[0], dat_answer[0]]) < 0.55:
        text = "Warning"
        start_alarm()

        color = (0, 255, 255)  # жёлтый
        base_color = (30, 30, 0)
        verdict = [e for i, e in [answer_ecg, vis_answer, dat_answer] if e != None]

    else:
        text = "Dangerous!"
        start_alarm()

        color = (0, 0, 255)  # красный
        base_color = (20, 0, 0)
        verdict = [e for i, e in [answer_ecg, vis_answer, dat_answer] if e != None]

    # Мягкое мигание для предупреждения и опасности
    if max([answer_ecg[0], vis_answer[0], dat_answer[0]]) >= 0.6:
        alphaq += dir_alpha * 0.05
        if alphaq > 1 or alphaq < 0:
            dir_alpha *= -1
        overlay = frame.copy()
        overlay[:] = color
        cv2.addWeighted(overlay, alphaq * 0.5, frame, 1 - alphaq * 0.5, 0, frame)
    else:
        frame[:] = base_color


    # Отображение текста
    meadly_text(text, 150, color)
    #cv2.putText(frame, text, (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5, cv2.LINE_AA)
    meadly_text(f"probability: {float(max([answer_ecg[0], vis_answer[0], dat_answer[0]])):.2f}", 220, (200, 200, 200))
    #cv2.putText(frame, f"probability: {float(max([answer_ecg[0], vis_answer[0], dat_answer[0]])):.2f}", (250, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    #cv2.putText(frame, f"{verdict}", (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    meadly_text(f"{verdict}", 80, (200, 200, 200))
    print()
    print(dat)
    print()
    if dat_answer!=None and dat != 'No rule variant satisfied. Consider more detailed analysis or clinician review.' and dat!=None:
        cn = 0
        for x, y in dat.items():
            cv2.putText(frame, f'{x} - {y[0]} * {y[1]:.2f}', (50, 380+cn), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255, 255), 2)
            cn += 30
    print(answer_ecg)
    if answer_ecg[1]!=None:
        # Загружаем изображение с альфа-каналом (PNG с прозрачностью)
        overlay = cv2.imread(f'card_for_code/plot_{1}.png', cv2.IMREAD_UNCHANGED)  # Важно: UNCHANGED!
      #  overlay = cv2.resize(overlay, (600, 300), interpolation=cv2.INTER_AREA)

        # Проверяем, что есть 4 канала (BGR + Alpha)
        if overlay.shape[2] == 4:
            # Разделяем каналы
            bgr = overlay[:, :, :3]  # BGR часть
            alpha = overlay[:, :, 3]  # Альфа-канал (0-255)

            # Нормализуем альфа-канал в диапазон [0, 1]
            alpha = alpha.astype(float) / 255.0

            # Размеры
            h, w = overlay.shape[:2]
            y, x = 250, 300  # Позиция, куда вставляем (можно менять)

            # Проверяем, не выходит ли за границы фона
            if y + h > frame.shape[0] or x + w > frame.shape[1]:
                print("Изображение выходит за границы фона!")
            else:
                # Область фона, куда будем вставлять
                roi = frame[y:y + h, x:x + w]

                # Альфа-блендинг: result = (alpha * foreground) + ((1 - alpha) * background)
                for c in range(3):  # по каждому каналу B, G, R
                    roi[:, :, c] = (alpha * bgr[:, :, c] + (1 - alpha) * roi[:, :, c])

                # Вставляем обратно в фон
                frame[y:y + h, x:x + w] = roi
        else:
            h, w = overlay.shape[:2]

            frame[100:100 + h, 300:300 + w] = overlay



    cv2.imshow("Monitor", frame)
    key = cv2.waitKey(100)
    if key == 27:  # ESC
        break
    if key == 32:

        if flag == False:
            flag = True
        else:
            flag = False
 

cv2.destroyAllWindows()