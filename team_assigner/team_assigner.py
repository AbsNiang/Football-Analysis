from sklearn.cluster import KMeans
class TeamAssigner:
    def __init__(self):
        self.team_colours = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self, image):
        # Reshape image to 2D array
        image_2d = image.reshape(-1, 3)

        # perform k-means clustering w 2 clusters
        kmeans = KMeans(n_clusters=2,init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans


    def get_player_colour(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[0: int(image.shape[0]/2), :]

        # Get clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # reshape the labels into the original image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        background_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - background_cluster

        #! may need to change from HSV to RGB
        player_colour = kmeans.cluster_centers_[player_cluster]

        return player_colour


    def assign_team_colour(self, frame, player_detections):
        player_colours = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_colour = self.get_player_colour(frame, bbox)
            player_colours.append(player_colour)

        kmeans = KMeans(n_clusters=2,init="k-means++", n_init=1)
        kmeans.fit(player_colours)

        self.kmeans = kmeans

        self.team_colours[1] = kmeans.cluster_centers_[0]
        self.team_colours[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_colour = self.get_player_colour(frame, player_bbox)

        team_id = self.kmeans.predict(player_colour.reshape(1, -1))[0]
        team_id += 1

        self.player_team_dict[player_id] = team_id

        return team_id