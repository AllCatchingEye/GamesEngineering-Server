# Übersicht

Nachdem Schafkopf ein Spiel für vier Spieler ist, müssen bis zu drei Spieler durch einen Bot ersetzt werden.

Dieser Bot muss dabei zwei Aufgaben erledigen:
 1. Entscheiden ob und was er spielen möchte (Spiel-Entscheidung)
 2. Die Partie mit den gegebenen Karten gegen die anderen Spieler ausspielen (Spiel-Partie)

# Umsetzung v1

## Spiel-Entscheidung

Die Spiel-Entscheidung ist ein Klassifikationsproblem. 

In dieser Version setzen wir für die Wahl der Spiel-Entscheidung ein Deep-Neural-Netz an. Grundlage hierfür bildet die Datenaufbereitung des Projekts des Wintersemesters 2022/23.

Langfristig gilt es noch zu evaluieren, ob hierfür ein Reinforcement-Algorithmus besser geeignet ist. Die Entscheidung, ob und was gespielt werden soll hängt damit wesentlich vom spielenden Agenten und nicht von externen Quellen ab.

### Voraussetzungen

Die dafür nötigen Voraussetzungen sind gegeben. Es existieren eine Menge an schon gespielten Partien sowie deren Ausgang, auf Basis dessen das Deep-Neural-Net ein passendes Verständnis entwickeln kann.

### Verwendung

Für die Wahl einer Spiele-Entscheidung übergibt man dem Bot seine Karten auf der Hand. Darauf basierend entscheidet der Bot ob und was er spielen würde. 

## Spiel-Partie

Für die eigentliche Spiele-Partie setzen wir einen Reinforcement-Algorithmus an. Dafür sehen wir folgende Gründe. Es ist sehr schwer, für alle möglichen Zustände in einer Schafkopf Partie einen passenden Algorithmus zu entwickeln. Unüberwachtes Lernen bietet den Vorteil, für einen Großteil der möglichen Zustände selbständig die besten Aktionen zu lernen. Das impliziert auch, dass der Agent besser werden kann als schon existierende Spieler.

### Voraussetzungen

Die dafür notwendigen Voraussetzungen sind auch gegeben. Es handelt sich um eine simulierbare Umgebung, dass keine Abhängigkeiten bestehen, die nicht abgebildet werden können. 

### Herausforderungen

Eine wesentliche Herausforderung wird der Lern-Prozess des Agenten sein. Es handelt sich hierbei um ein Multi-Agent-Environment, sodass zeitgleich mehrere Agenten mit oder gegeneinander spielen (Mix aus "fully competitive" and "fully cooperative"). Hierfür gilt es einen guten Ansatz zu finden.

Zudem gilt es einen passenden Reward zu finden. Prinzipiell ist es das ultimative Ziel einer Partie, diese zu gewinnen. Daraus nun jedoch den Reward für die einzelnen Entscheidungen (das Legen der acht Karten) abzuleiten, erscheint initial schwer. 

### Verwendung

Die Nutzung des Bots erfolgt zustandslos. Dafür übergibt man folgende Informationen: 

- den Typ der Spiele-Partie (Sauspiel, Wends, Geier, Ramsch, ...)
- die schon gespielten Karten (Typ der Karte + Spieler der Karte)
- die Karten des Bots auf der Hand

Der Bot entscheidet sich darauf basierend für seinen eigenen nächsten Zug. 
