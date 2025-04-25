BEGIN TRANSACTION;

DELETE 
  FROM public.total_stat
 WHERE total_stat.date >= CURRENT_DATE - 30
;

INSERT INTO public.total_stat (
    project_name,
    date,
    deposit_usd,
    withdrawal_usd
)
WITH _projects AS (
    SELECT project.projec_id,
           project.name AS project_name
      FROM reference.project
)
  SELECT _projects.project_name,
         user_daily_kpi.date AS date,
         SUM(user_daily_kpi.deposit_usd) AS deposit_usd,
         SUM(user_daily_kpi.withdrawal_usd) AS withdrawal_usd
    FROM public.user_daily_kpi
         INNER JOIN _projects
         ON _projects.projec_id = user_daily_kpi.projec_id
   WHERE user_daily_kpi.date >= CURRENT_DATE - 30
GROUP BY 1, 2
;

END TRANSACTION;