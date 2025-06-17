CREATE TABLE simple.create_table_as_select_from_table AS
SELECT id AS user_id,
       name
  FROM source.users;