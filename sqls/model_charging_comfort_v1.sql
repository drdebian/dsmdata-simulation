with cte0 as
(SELECT 
 vehicle, period
from modelev_perfect b
where b.loadend = 1
  and b.label like "%period00"
  and runningdate > (select max(timestamp) from lastrun)
group by vehicle, period
)

select 
 replace(substr(b.label, 1, length(b.label) - 2), "_period", "") as config
,a.model
,a.pvprc
,a.bessprc
,cast(substr(b.label, length(b.label) - 1, 2) as int) as period
,b.vehicle
,b.EVSOC
from result2_direct b
inner join result0_direct a
       on a.label = b.label
	  and a.runningdate = b.runningdate
inner join cte0 c
       on c.vehicle = b.vehicle
	  and c.period = cast(substr(b.label, length(b.label) - 1, 2) as int)
where b.period = 0
  and b.runningdate > (select max(timestamp) from lastrun)

UNION

select 
 replace(substr(b.label, 1, length(b.label) - 2), "_period", "") as config
,a.model
,a.pvprc
,a.bessprc
,cast(substr(b.label, length(b.label) - 1, 2) as int) as period
,b.vehicle
,b.EVSOC
from result2_rule b
inner join result0_rule a
       on a.label = b.label
	  and a.runningdate = b.runningdate
inner join cte0 c
       on c.vehicle = b.vehicle
	  and c.period = cast(substr(b.label, length(b.label) - 1, 2) as int)
where b.period = 0
  and b.runningdate > (select max(timestamp) from lastrun)

UNION

select 
 replace(substr(b.label, 1, length(b.label) - 2), "_period", "") as config
,a.model
,a.pvprc
,a.bessprc
,cast(substr(b.label, length(b.label) - 1, 2) as int) as period
,b.vehicle
,b.EVSOC
from result2_pred b
inner join result0_pred a
       on a.label = b.label
	  and a.runningdate = b.runningdate
inner join cte0 c
       on c.vehicle = b.vehicle
	  and c.period = cast(substr(b.label, length(b.label) - 1, 2) as int)
where b.period = 0
  and b.runningdate > (select max(timestamp) from lastrun)

UNION

select 
 replace(substr(b.label, 1, length(b.label) - 2), "_period", "") as config
,a.model
,a.pvprc
,a.bessprc
,cast(substr(b.label, length(b.label) - 1, 2) as int) as period
,b.vehicle
,b.EVSOC
from result3_stoch b
inner join result0_stoch a
       on a.label = b.label
	  and a.runningdate = b.runningdate
inner join cte0 c
       on c.vehicle = b.vehicle
	  and c.period = cast(substr(b.label, length(b.label) - 1, 2) as int)
where b.period = 0
  and b.scenario = 0
  and b.runningdate > (select max(timestamp) from lastrun)
  
UNION

select 
 replace(substr(b.label, 1, length(b.label) - 2), "_period", "") as config
,a.model
,a.pvprc
,a.bessprc
,cast(substr(b.label, length(b.label) - 1, 2) as int) as period
,b.vehicle
,b.EVSOC
from result2_perfect b
inner join result0_perfect a
       on a.label = b.label
	  and a.runningdate = b.runningdate
inner join cte0 c
       on c.vehicle = b.vehicle
	  and c.period = cast(substr(b.label, length(b.label) - 1, 2) as int)
where b.period = 0
  and b.runningdate > (select max(timestamp) from lastrun)

  
