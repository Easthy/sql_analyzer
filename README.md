#### The goal of this project 
The goal of this project is to build dependency graphs between tables and columns in order to easily track how changes in one of the parent models affect the downstream (child) models.  
Ideally, this could also be used to automatically generate tests.

#### Why
Existing solutions like dbt and sqlmesh impose certain limitations — while they do solve the problem of dependecies (but they are not able to visualize), they require integration into the project (a lot of rework) and introduces additional constraints (for example, you may no longer be able to write raw DDL when you really need it), or new issues. 

### How to Use
You should place all source table definitions into a `source.yml` file.  
The model's name will be parsed from the file name, which must follow the template `schema.table_name.sql`.  
Run the `sql_analyzer.py` script; the result will be saved as `dependency_state.json`. Then, you can execute `draw_graphviz.py` or `draw_graph.py` to generate graphs in `dependency_graph.png` and `graph_output.png` files, respectively

#### An example of the graph is shown in graph_output.png (draw_graphviz.py)
![alt_text](https://github.com/Easthy/sql_analyzer/blob/main/dependency_graph.png)

#### An example of the graph is shown in graph_output.png (draw_graph.py)
![alt text](https://github.com/Easthy/sql_analyzer/blob/main/graph_output.png)

#### Issues
- There is a problem if the DISTKEY is defined right after a column's data type (sqlglot throws an error). It should be defined after all columns
- Only direct dependencies will be found. For example, if your column col_1 does not directly depend on col_2, but you use col_2 to filter rows, then col_2 will not be considered its ancestor.
