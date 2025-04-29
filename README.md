### The goal of this code is to build dependency graphs between tables (and ideally columns) in order to easily track how changes in one of the parent models affect the downstream (child) models. Ideally, this could also be used to automatically generate tests.

#### Existing solutions like dbt and sqlmesh impose certain limitations — while they do solve this problem, they require integration into the project (a lot of rework), which introduces additional constraints (for example, you may no longer be able to write raw DDL when you really need it), or new issues (sqlmesh is still somewhat immature, unfortunately).

#### An example of the graph is shown in graph_output.png.
![alt text](https://github.com/Easthy/sql_analyzer/blob/main/graph_output.png)

#### Issues
- There is a problem if the DISTKEY is defined right after a column's data type (sqlglot throws an error). It should be defined after all columns
- There is a problem handling temporary table dependencies if they are used to build the model