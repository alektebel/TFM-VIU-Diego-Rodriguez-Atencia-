

import os

lista_de_objetos = os.listdir('/home/diego/TFM/GRAB/')


for file in lista_de_objetos:
    if ".stl" in file:

        object_mesh = '''<mujoco>
          <asset>
            <mesh name="{0}" file="/home/diego/TFM/GRAB/{1}"/>
          </asset>
          <worldbody>
            <body name="{0}" mocap="true">  <!-- mocap enables manual positioning -->
              <geom type="mesh" mesh="{0}"/>
            </body>
              </worldbody>
        </mujoco>'''.format(file.replace(".stl", "_mesh"), file)
        #print(object_mesh)
        # Save to XML file
        with open("./objects/"+file.replace(".stl", "_scene.xml"), 'w') as f:
            f.write(object_mesh)


