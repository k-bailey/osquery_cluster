# osquery_cluster_data


This code clusters hosts together based on osquery data. **It is a work in progress.**
The first project is an attempt to cluster hosts together based on installed packages, crontab entries and installed kernel modules using kmeans. Ideally this can be expanded to take data from many osquery tables and correlate it into a host profile that can be used to cluster hosts and identify outliers.

## Basic Usage

Data must be formatted in a csv and then fed as the first argument to the script. Data needs to be in the following format:

| host | os_major | os_minor | package | kernel_module | cron_command |
| ---- | ---- | ---- | ---- | ---- | ---- |
| host1 | 7 | 0 | pkg1\|pkg2\|pkg3\|...\|pkgN | km1\|km2\|km3\|...\|kmN | cron1\|cron2\|cron3\|...\|cronN |
| host2 | 6 | 9 | pkg1\|pkg2\|pkg3\|...\|pkgN | km1\|km2\|km3\|...\|kmN | cron1\|cron2\|cron3\|...\|cronN |
| ... | ... | ... | ... | ... | ... |


The second argument adjusts the number of clusters kmeans will use. `python host_cluster.py -h` for more.

Usage:
```python
 python host_cluster.py <data> <clusters>
```

## Results

Initial results have been positive, clones of similar hosts are being placed in the same  cluster and hosts with different installed software are being placed in their own cluster.
