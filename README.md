# Wireless Rechargeable Sensors Network
##Q-charging 
- Implementation of a Q_learning based charging strategy with Kmeans network-clustering on WRSN with multiple mobile chargers.



## Experiments:


```bash
$ python Test.py
experiment_type:
experiment_index:
```

| Experiment_index      Experiment_type|    0    |    1    |    2    |    3     |    4   |
|--------------------------------------|---------|---------|---------|----------|--------|
| **node**                             |   700   |   800   | __900__ |   1000   |   1100 |
| **target**                           |   500   |   550   | __600__ |   650    |   700  |
| **MC**                               |   2     | __3__   |   4     |   5      |   6    |
| **prob**                             |   0.5   | __0.6__ |   0.7   |   0.8    |   0.9  |
| **package**                          | __500__ |   550   |   600   |   650    |   700  |
| **cluster**                          |   40    |   50    |   60    |   70     | __80__ |

> `target` experiments must be *reconstructed* to match `node` experiments range if modified 
## Results:

- All experiment results are updated at this [sheet](https://husteduvn-my.sharepoint.com/:x:/g/personal/long_nt183586_sis_hust_edu_vn/EVypWNIGoz1GkK7v6QYDmccBJKAzweAXJr8ZhFF94kYgnw?e=Jrwb9k).

## Requirements:
- `pandas==1.1.3`  
- `scipy==1.5.2`    
- `numpy==1.19.2`
- `scikit_learn==0.24.2`
