BEGIN TRANSACTION;


-- Calculate last user's activity
DROP TABLE IF EXISTS user_last_activity_tmp
;
CREATE TEMPORARY TABLE user_last_activity_tmp (
    user_hash_key          CHAR(32),
    last_activity          TIMESTAMP ENCODE AZ64,
    last_activity_apk      TIMESTAMP ENCODE AZ64
)
    DISTSTYLE KEY
    DISTKEY (user_hash_key)
    SORTKEY (last_activity)
;

INSERT INTO user_last_activity_tmp(
    user_hash_key,
    last_activity,
    last_activity_apk
)
  SELECT event_tracker.user_hash_key,
         MAX(event_tracker.date) AS last_activity,
         MAX(
            CASE
                WHEN event_tracker.user_agent = 'mobile-application' THEN lnk_user_mwl_event.date
            END
         ) AS last_activity_apk
    FROM mwl.event_tracker
GROUP BY event_tracker.user_hash_key
;

CREATE TEMPORARY TABLE user_diff_tmp (
    user_hash_key          CHAR(32),
    last_activity          TIMESTAMP ENCODE AZ64,
    last_activity_apk      TIMESTAMP ENCODE AZ64
)
    DISTSTYLE KEY
    DISTKEY (user_hash_key)
;

INSERT INTO user_diff_tmp
SELECT users.user_hash_key,
       user_last_activity_tmp.last_activity AS last_activity,
       user_last_activity_tmp.last_activity_apk AS last_activity_apk,
  FROM mwl.mysql_users AS users

       LEFT JOIN user_last_activity_tmp
       ON user_last_activity_tmp.user_hash_key = users.user_hash_key
;

-- Upsert
DELETE FROM business_vault.user_attributes
      USING user_diff_tmp
      WHERE user_attributes.user_hash_key = user_diff_tmp.user_hash_key
;
INSERT INTO business_vault.user_attributes(user_hash_key, calc_last_activity, calc_last_activity_apk)
     SELECT DISTINCT user_hash_key, fake, last_activity AS calc_last_activity, last_activity_apk AS calc_last_activity_apk
       FROM user_diff_tmp
;
DROP TABLE IF EXISTS user_diff_tmp
;


END TRANSACTION;