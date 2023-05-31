import mysql.connector
import creds

#creating Mysql database ... Initialise this file only once
db = mysql.connector.connect(
    host=creds.HOST,
    user=creds.USERNAME,
    passwd=creds.PASSWORD,
    database=creds.DATABASE
    )

mycursor = db.cursor()

clasing = ['Asuka Langley Soryu',                           
 'Chitoge Kirisaki',
 'Kosaki Onodera',
 'Kurumi Tokisaki',
 'Rei Ayanami',
 'Saber Fate',
 'Tobichi Origami',
 'Tohsaka Rin']

anime_titles = ['Neon Genesis Evangelion', 'Nisekoi', 'Nisekoi', 'Date A Live', 'Neon Genesis Evangelion', 'Fate/stay night', 'Date A Live', 'Fate/stay night']

character_ages = [18, 17, 16, 19, 14, 25, 20, 18]

character_heights = [155, 160, 157, 157, 149, 154, 152, 159]

character_weights = [58, 52, 50, 55, 47, 60, 54, 53]

character_descriptions = [
    "Asuka Langley Soryu: A fiery and strong-willed pilot from Neon Genesis Evangelion, Asuka displays exceptional skills as an Eva pilot, but her troubled past and complex personality make her prone to emotional outbursts.",
    "Chitoge Kirisaki: A high-spirited girl from Nisekoi, Chitoge becomes involved in a fake relationship. Despite initial clashes, she develops genuine feelings, showcasing her caring side and loyalty.",
    "Kosaki Onodera: Sweet and gentle, Kosaki harbors a secret crush on the main character in Nisekoi. Her kind-hearted nature and innocence make her a beloved character in the series.",
    "Kurumi Tokisaki: A captivating and mysterious character from Date A Live, Kurumi possesses a dark past. With her striking appearance and crimson eyes, she displays a manipulative and cunning nature.",
    "Rei Ayanami: An enigmatic character from Neon Genesis Evangelion, Rei is quiet, introverted, and emotionally distant. Her origins and purpose are gradually unveiled throughout the series.",
    "Saber Fate: Also known as Artoria Pendragon, Saber is a noble and stoic character in the Fate/stay night series. Her unwavering dedication and strong sense of justice make her a respected presence.",
    "Tobichi Origami: Initially portrayed as a cool and level-headed girl from Date A Live, Tobichi becomes deeply involved in supernatural events. Her character arc delves into personal struggles and conflicting emotions.",
    "Tohsaka Rin: A prominent character in the Fate/stay night series, Rin possesses exceptional magical talent. Determined and ambitious, she showcases her growth and unwavering pursuit of her goals."
]


Q1 = "CREATE TABLE Anime(name VARCHAR(30) PRIMARY KEY,anime VARCHAR(30) NOT NULL, age INT NOT NULL, height INT NOT NULL, weight INT NOT NULL, description VARCHAR(300) NOT NULL)"
mycursor.execute(Q1)

for i in range(8):
    mycursor.execute("INSERT INTO Anime (name, anime, age, height, weight, description)  VALUES (%s,%s,%s,%s,%s,%s)", (clasing[i], anime_titles[i], character_ages[i], character_heights[i], character_weights[i], character_descriptions[i]))
db.commit()

