NTU FYP: RobotX Development of Surface Vehicle's Visual Perception of Dynamic Features
==================================================
This project focuses on the development of a visual perception system for an Unmanned Surface Vehicle (USV), aimed at generating a sequence of dynamic features such as colors. The project is undertaken with the aim of completing Task 5 of the Maritime RobotX Challenge 2024.

This project entails the integration of hardware and development of software for the Monocular Vision System, which is designed to perform localization and color recognition of LED Panel to produce the sequence of colors. Testing of the Monocular Vision System is
done by the integration of hardware and software of the LED Panel. Furthermore, a Graphical User Interface (GUI) is designed to facilitate the process of the system.

The final section of this project will outline the results of the Monocular Vision System performance both indoors and outdoors, along with potential future improvements for further development.

Hardware
--------
Designed a test rig since it will not be mounted on the USV.

* 1 x Arducam 64MP Camera https://amicus.com.sg/index.php?route=product/product&manufacturer_id=79&product_id=8222

* 1 x Raspberry Pi 4B 2GB

Isometric and Front View of Monocular Vision System

![image](https://github.com/user-attachments/assets/ec0bbe02-4f32-4674-9bef-bdd7759dae7d)

Software
--------
The main difference between _rasp.py and _windows.py code lies in their image capture functions. _rasp.py is tailored for compatibility with the Arducam 64MP, while _windows.py has been tested using the Logitech C920.

Note: Change the folder path in the code to your desired destination.

Images
--------
![bbb](https://github.com/jonxjonx/Monocular_Vision_RobotX/assets/142519124/4e8678d8-35cc-4fdf-b0fa-ef7101fa3890)

![aaa](https://github.com/jonxjonx/Monocular_Vision_RobotX/assets/142519124/7597d187-1b80-49d6-a6b8-9df1824ddeb5)

![woo1](https://github.com/jonxjonx/Monocular_Vision_RobotX/assets/142519124/7a061e4a-10d2-4d0e-b556-2d586636acdf)

![woo2](https://github.com/jonxjonx/Monocular_Vision_RobotX/assets/142519124/1fc47436-8af4-4a69-a62a-610d8a842883)
