BEGIN TRANSACTION;

CREATE TEMPORARY TABLE user_slice (
    user_hash_key CHAR(32),
    user_id INTEGER ENCODE AZ64,
    project_id INTEGER ENCODE AZ64,
    registration_date TIMESTAMP ENCODE DELTA,
    fake INTEGER ENCODE AZ64,
    email VARCHAR ENCODE ZSTD,
    gender INTEGER ENCODE AZ64
)
    DISTSTYLE KEY
    DISTKEY(user_hash_key)
    SORTKEY(registration_date)
;

INSERT INTO user_slice(
    user_hash_key,
    user_id,
    project_id,
    registration_date,
    fake,
    email,
    gender
)
SELECT users.user_hash_key,
       users.user_id,
       users.project_id,
       users.registration_date,
       users.fake,
       users.email,
       users.gender
  FROM public.users
 WHERE users.registration_date >= CURRENT_DATE - 30
;

DELETE 
  FROM public.users
 USING user_slice
 WHERE user_slice.user_hash_key = users.user_hash_key
;

CREATE TEMPORARY TABLE user_source AS
SELECT users.user_hash_key,
       users.user_id,
       users.project_id,
       users.fake,
       users.gender,
       users.email,
       users.registration_date
  FROM mwl.mysql_users AS users
 WHERE users.project_id NOT IN (5, 7)
       AND users.registration_date >= CURRENT_DATE - 30
;

INSERT INTO public.users (
    user_hash_key,
    user_id,
    project_id,
    registration_date,
    fake,
    email,
    gender
)
WITH _user_rewrite AS (
    SELECT user_source.user_hash_key,
           user_source.user_id,
           CASE
                WHEN user_source.project_id IN (2, 3)
                     THEN 1
                ELSE user_source.project_id
           END AS project_id,
           user_source.fake,
           user_source.gender,
           user_source.email,
           user_source.registration_date
      FROM user_source
)
SELECT _user_rewrite.user_hash_key,
       _user_rewrite.user_id,
       _user_rewrite.project_id,
       _user_rewrite.registration_date,
       _user_rewrite.fake,
       _user_rewrite.email,
       _user_rewrite.gender
  FROM _user_rewrite
;

END TRANSACTION; 
