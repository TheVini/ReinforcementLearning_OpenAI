## Computer and General Spec
- CPU: i5-8400 (with Water Cooler)
- RAM: 1x16 Gb 2133Mhz DDR4 
- GeForce GTX 1060 6GB
- Cuda 10.1
- PyCharm as IDE

#### [1. OpenAI Gym - LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/)

<table align="center">
    <thead>
        <tr>
            <th>Topic</th>
            <th>Note</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center">Definition of success</td>
            <td align="center">Average score of at least 200 points for the last 150 episodes.</td>
        </tr>
        <tr>
            <td align="center">Training duration</td>
            <td align="center">1h 51m 28s</td>
        </tr>
        <tr>
            <td align="center">Technique</td>
            <td align="center">Deep Q-Learning</td>
        </tr>
        <tr>
            <td align="center">Smallest loss</td>
            <td align="center">0.183936 (using Huber loss)</td>
        </tr>
    </tbody>
</table>

<table align="center">
    <thead>
        <tr>
            <th>Before training</th>
            <th>After training</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center">
                <img src="https://github.com/TheVini/DeepReinforcement_OpenAI/blob/master/Others/lunarlanderbefore.gif" width="300" height="200">
            </td>
            <td align="center"> 
                <img src="https://github.com/TheVini/DeepReinforcement_OpenAI/blob/master/Others/lunarlanderafter.gif" width="300" height="200">
            </td>
        </tr>
    </tbody>
</table>

#### [2. OpenAI Gym - MountainCar-v0](https://gym.openai.com/envs/MountainCar-v0/)

<table align="center">
    <thead>
        <tr>
            <th>Topic</th>
            <th>Note</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center">Definition of success</td>
            <td align="center">Average score of at least -110 points for the last 100 episodes.</td>
        </tr>
        <tr>
            <td align="center">Training duration</td>
            <td align="center">17m 48s</td>
        </tr>
        <tr>
            <td align="center">Technique</td>
            <td align="center">Deep Q-Learning</td>
        </tr>
        <tr>
            <td align="center">Smallest loss</td>
            <td align="center">0.551261 (using MSE)</td>
        </tr>
    </tbody>
</table>

<table align="center">
    <thead>
        <tr>
            <th>Before training</th>
            <th>After training</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center">
                <img src="https://github.com/TheVini/DeepReinforcement_OpenAI/blob/master/Others/mountaincarbefore.gif" width="300" height="200">
            </td>
            <td align="center"> 
                <img src="https://github.com/TheVini/DeepReinforcement_OpenAI/blob/master/Others/lunarlanderafter.gif" width="300" height="200">
            </td>
        </tr>
    </tbody>
</table>