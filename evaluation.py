import os
import networkx as nx
import matplotlib.pyplot as plt
import re
import cv2
import shutil
from datetime import datetime
import imagecollage

# Set evaluation folder
basepath = os.path.dirname(os.path.realpath(__file__))
evaluate = os.path.join(basepath, 'evaluate_matches')

print()

### USER INPUT

reset = input("""Select operation:
Enter '1' - Initial run or reset
Enter '2' - Revise components by selecting component specific cut-offs
Enter '3' - Finalise: create component folders and subgraphs
(WARNING: finalising will entirely re-establish the component structure)

""")

if int(reset) == 2:

    print()

    allfolders = input('Restrict all components? Y/N ')

    if 'n' in allfolders.lower():

        componenttorestrict = str(input("Enter component number(s) to retrict matches (separated by comma's): "))
        componenttorestrict = componenttorestrict.split(',')
        componenttorestrict = [int(i) for i in componenttorestrict]
        print('Confirming components to be restricted',componenttorestrict)

        matchparameters = []

        print()

        for i in componenttorestrict:
            restrict = int(input('Set minimum number of feature matches for component '+str(i)+': '))
            matchparameters.append(int(restrict))

    if 'y' in allfolders.lower():
        matchpar = int(input('Set minimum of matches: '))
        componenttorestrict = []

### GET SOURCE FILES

files = []

filepath = os.path.join(basepath, "source_images_clean")

for r, d, f in os.walk(filepath):
    for file in f:
        if '.jpg' in file:
            files.append(file)

### GET MATCH FILES

# NEW RUN
if int(reset) == 1:
    filepath = os.path.join(basepath, "matches")

# REVISION RUN

if int(reset) == 2:

    filepath = evaluate

    for r, d, f in os.walk(filepath):
        for file in f:
            if '.jpg' in file:

                if 'y' in allfolders.lower():

                    if matchpar > int(file.replace('.jpg','').split('_')[2]):
                        removefile = os.path.join(r, file)
                        print('Removing match file:',removefile)
                        os.remove(removefile)

                if int(r.split('comp')[1]) in componenttorestrict:

                    index = componenttorestrict.index(int(r.split('comp')[1]))

                    if matchparameters[index] > int(file.replace('.jpg','').split('_')[2]):
                        removefile = os.path.join(r, file)
                        print('Removing match file:',removefile)
                        os.remove(removefile)

# FINAL RUN
if int(reset) == 3:
    filepath = os.path.join(basepath, "evaluate_matches")
    final_sub = os.path.join(evaluate, 'finalised_solution')

    try:
        shutil.rmtree(final_sub)
        print()
    except:
        pass

###

matchfiles = []
matches = []

for r, d, f in os.walk(filepath):
    for file in f:
        if '.jpg' in file:
            matchfiles.append(file)

            file1 = file.split('_')[0]
            file2 = file.split('_')[1]

            matches.append((file1,file2))

### Empty folder and populate on first run

if int(reset) == 1:

    try:
        shutil.rmtree(evaluate)
        print()
    except:
        pass

    try:
        os.mkdir(evaluate)
    except:
        pass

### Compile graphs

G=nx.Graph()
G.add_edges_from(matches)

### Count nodes in network

networknodes = []
components = (G.subgraph(c) for c in nx.connected_components(G))

for component in components:
    for node in component:
        networknodes.append(node)

networknodes = list(set(networknodes))

### Entire procdure:

if int(reset) == 1 or int(reset) == 3:

    # Run through components
    components = (G.subgraph(c) for c in nx.connected_components(G))

    for index,component in enumerate(components):

        # New run make folders with match files

        if int(reset) == 1:

            evaluate_sub = os.path.join(evaluate, 'comp'+str(index+1))

            filename = os.path.join(evaluate_sub,'Comp'+str(index+1)+".pdf")

            try:
                os.mkdir(evaluate_sub)
            except:
                pass

            nx.draw(component,node_size=25,node_color='black')
            plt.savefig(filename)
            plt.clf()

            print('Component',index+1,'-',len(component),'nodes')

            for node in component:
                #print(node)
                networknodes.append(node)

            #print()

            for matchfile in matchfiles:
                if matchfile.split('_')[0] in component and matchfile.split('_')[1] in component:

                    fileloc = os.path.join(basepath, "matches", matchfile)
                    #print(fileloc)
                    image = cv2.imread(fileloc)

                    targetloc = os.path.join(evaluate_sub, matchfile)
                    #print(targetloc)
                    cv2.imwrite(targetloc,image)

        # Final run make files with

        if int(reset) == 3:

            comp_sub = os.path.join(final_sub, 'comp'+str(index+1))

            try:
                os.mkdir(final_sub)
            except:
                pass

            filename1 = os.path.join(final_sub,'Comp'+str(index+1)+".pdf")
            filename2 = os.path.join(final_sub,'Comp'+str(index+1)+".jpg")

            nx.draw(component,node_size=25,node_color='black')
            plt.savefig(filename1)
            plt.savefig(filename2)
            plt.clf()

            print('Component',index+1,'-',len(component),'nodes')

            componentimages = []
            collagename = os.path.join(final_sub,'component'+str(index+1)+'.jpg')

            for node in component:

                try:
                    os.mkdir(comp_sub)
                except:
                    pass

                fileloc = os.path.join(basepath, "source_images", node+'.jpg')
                #print(fileloc)
                image = cv2.imread(fileloc)

                targetloc = os.path.join(comp_sub, node+'.jpg')
                #print(targetloc)
                cv2.imwrite(targetloc,image)

                componentimages.append(targetloc)

            imagecollage.make_collage(componentimages,collagename)

            image1 = cv2.imread(filename2).astype("float32")
            image2 = cv2.imread(collagename).astype("float32")

            print(type(image1))
            print(type(image2))

            def vconcat_resize(img_list, interpolation  = cv2.INTER_CUBIC):

                # take minimum width
                w_min = min(img.shape[1]
                            for img in img_list)

                # resizing images
                im_list_resize = [cv2.resize(img,
                                  (w_min, int(img.shape[0] * w_min / img.shape[1])),
                                             interpolation = interpolation)
                                  for img in img_list]
                # return final image
                return cv2.vconcat(im_list_resize)

            # function calling
            v_img = vconcat_resize([image1, image2])

            cv2.imwrite(os.path.join(final_sub,'Result'+str(index+1)+'.jpg'),v_img)


### STATS
print('\nTotal number of original images',len(files))
print('Total number of nodes in the network',len(networknodes))
print('Percentage images as nodes:',str(round(len(networknodes)/len(files)*100,2))+'%')
print('Number of components:',nx.number_connected_components(G))

nx.draw(G,node_size=10,node_color='black')
filename = os.path.join(basepath,"Graph"+str(datetime.now())+".pdf")
plt.savefig(filename, format="PDF")
