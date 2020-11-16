import numpy as np

def find_corners(image_size, tile_size, network_size, overlap=0):
    #Do the math to find the corners of the tiles:
    #Set up variables for iteration
    topCorner = [0,0]
    shift = False
    corners = []
    #Easily track incomplete squares
    size = [network_size[0], network_size[1]]

    while True:
        while not shift:

            bottomCorner = list(map(lambda x,y:x+y, topCorner, tile_size)) #Make a full tile
    
            if bottomCorner[0] > image_size[0]: #If the full tile bottom is off the image
                bottomCorner[0] = image_size[0] #Stop to the tile bottom at the image edge
        
            if bottomCorner[1] > image_size[1]: #If the right egde of the tile is off the image
                bottomCorner[1] = image_size[1] #Stop the right edge of the tile at the image edge
                shift = True #Time to shift to the next y position
        
            size = list(map(lambda x,y:x-y, bottomCorner, topCorner)) #Record size of actual image in this tile
    
            corners += [np.array([topCorner, bottomCorner, size])] #Add this into to our list of corners
    
            topCorner[1] += tile_size[1] - overlap #Move top corner x-position over
            
        #Shift to next y-position
        shift = False #Reset flag
            
        topCorner[0] += tile_size[0] - overlap #Change top corner y-position
    
        topCorner[1] = 0 #Change top corner x-position
    
        if bottomCorner == image_size: #Check to see if we are at the very bottom corner
            break #Stop if we are
            
    return corners

def create_tiles(image, corners, network_size):
    tiles_list = [] #List to hold tile arrays
    for cornerSet in corners:
        topY = cornerSet[0,0]
        topX = cornerSet[0,1]
        botY = cornerSet[1,0]
        botX = cornerSet[1,1]
        
        tile = image[topY:botY, topX:botX, :]
        if list(tile.shape) != network_size: #Need to ensure tiles are all the proper size for the network
            inter_array = np.zeros((np.array(network_size)))
            inter_array[0:tile.shape[0],0:tile.shape[1],:] = tile #Pad out the tile size with zeros
            tile = inter_array
        tiles_list += [tile]

    tilesArr = np.array(tiles_list)
        
    return tilesArr

def assemble_pred(predArr, corners, image, network_size):
    pred_image = np.zeros((image.shape[0], image.shape[1]))
    pred_image[:,:] = np.nan #NaN so we only average the overlap
    corn_idx = 0
    for pred in predArr:
        cornerSet = corners[corn_idx]
        topY = cornerSet[0,0]
        topX = cornerSet[0,1]
        botY = cornerSet[1,0]
        botX = cornerSet[1,1]
        tile_shape = cornerSet[2]

        img_tile = pred_image[topY:botY, topX:botX] #Tile from our total prediction image
        
        flat_img_tile = np.ndarray.flatten(img_tile) 

        pred = pred[0:tile_shape[0],0:tile_shape[1]]
        
        flat_pred_tile = np.ndarray.flatten(pred)
    
        new_img_tile = list(map(lambda x, y: y if np.isnan(x) else (x+y)/2, flat_img_tile, flat_pred_tile)) #Average the predictions if x isn't NaN

        new_img_tile = np.asarray(new_img_tile).reshape(tile_shape)
    
        pred_image[topY:botY, topX:botX] = new_img_tile
            
        corn_idx += 1  
    return pred_image      
