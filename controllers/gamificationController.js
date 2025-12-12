// controllers/gamificationAdvancedController.js
const User = require("../modules/userSchema");

// Configuration / weights (tweakable)
const XP_WEIGHTS = {
    PER_SESSION: 25,
    PER_QUESTION: 3,
    PER_STRONG_SKILL: 6,
    DOMAIN_MASTERY_MULTIPLIER: 10,
    CONSISTENCY_MULTIPLIER: 15,
    ENGAGEMENT_MULTIPLIER: 2,
    GROWTH_MULTIPLIER: 20,
    ACCURACY_MULTIPLIER: 1.5
};
const BASE_XP = 10; // baseline xp
const XP_PER_LEVEL = 100; // xp required per level tier

// Helpers
const safe = (v, fallback = 0) => (typeof v === "number" && !isNaN(v) ? v : fallback);
const avg = (arr) => (Array.isArray(arr) && arr.length ? arr.reduce((a, b) => a + safe(b), 0) / arr.length : 0);
const daysBetween = (a, b) => {
    const diff = Math.abs(new Date(a).setHours(0, 0, 0, 0) - new Date(b).setHours(0, 0, 0, 0));
    return Math.round(diff / (1000 * 60 * 60 * 24));
};

function calcLevelFromXP(xp) {
    return Math.floor(xp / XP_PER_LEVEL) + 1;
}

// Main controller
exports.getAdvancedGamification = async (req, res) => {
    try {
        const userId = req.user.id;
        const user = await User.findById(userId).lean();

        if (!user) return res.status(404).json({ success: false, message: "User not found" });

        const sessions = Array.isArray(user.interview_sessions) ? user.interview_sessions : [];
        const profileSkills = Array.isArray(user.skills) ? user.skills : [];
        const targetDomain = (user.target_domains && user.target_domains[0]) || null;

        // Aggregate raw counts
        const totalSessions = sessions.length;
        let totalQuestions = 0;
        let totalStrongSkills = 0;
        let totalWeakSkills = 0;
        const domainMap = new Map(); // domain => { count, avgScore }
        const domainScores = {}; // for response
        const sessionAverages = []; // overall_average per session (time-ordered by startedAt)
        const accuracyList = [];
        let engagementCount = 0; // mcq + coding + voice attempts sum
        const weakSkillOccurrences = {}; // skill => count
        const strongSkillSet = new Set();
        const activeDaysSet = new Set();
        let firstSessionDate = null;
        let lastSessionDate = null;

        // Sort sessions by startedAt ascending to compute growth/progression
        const sessionsSorted = sessions.slice().sort((a, b) => {
            const aa = a.startedAt ? new Date(a.startedAt) : new Date(0);
            const bb = b.startedAt ? new Date(b.startedAt) : new Date(0);
            return aa - bb;
        });

        sessionsSorted.forEach((s) => {
            // dates
            if (s.startedAt) {
                if (!firstSessionDate) firstSessionDate = new Date(s.startedAt);
                lastSessionDate = new Date(s.startedAt);
                activeDaysSet.add(new Date(s.startedAt).toDateString());
            }

            // questions answered
            const qAnswered = safe(s?.detailed_summary?.session_stats?.questions_answered, 0);
            totalQuestions += qAnswered;

            // engagement attempts
            engagementCount += safe(s?.detailed_summary?.session_stats?.mcq_attempted, 0);
            engagementCount += safe(s?.detailed_summary?.session_stats?.subjective_attempted, 0);
            engagementCount += safe(s?.detailed_summary?.session_stats?.voice_attempted, 0);
            engagementCount += safe(s?.detailed_summary?.session_stats?.coding_attempted, 0);

            // skill_analysis
            const strongArr = Array.isArray(s?.skill_analysis?.stronger_skills) ? s.skill_analysis.stronger_skills : [];
            const weakArr = Array.isArray(s?.skill_analysis?.weaker_skills) ? s.skill_analysis.weaker_skills : [];
            totalStrongSkills += strongArr.length;
            totalWeakSkills += weakArr.length;
            strongArr.forEach(sk => strongSkillSet.add(sk));
            weakArr.forEach(w => weakSkillOccurrences[w] = (weakSkillOccurrences[w] || 0) + 1);

            // domain mastery
            const d = s.domain || "unknown";
            const ovAvg = safe(s?.detailed_summary?.session_stats?.overall_average, null);
            if (!domainMap.has(d)) domainMap.set(d, { count: 0, scores: [] });
            const dm = domainMap.get(d);
            dm.count += 1;
            if (ovAvg !== null) dm.scores.push(ovAvg);

            // session averages list
            sessionAverages.push(ovAvg !== null ? ovAvg : 0);

            // accuracy
            if (typeof s.accuracy === "number") accuracyList.push(s.accuracy);
        });

        // compute domain mastery map
        domainMap.forEach((v, k) => {
            const domainAvg = avg(v.scores);
            domainScores[k] = +domainAvg.toFixed(3);
        });

        // DOMAIN MASTERY aggregate (weighted)
        const domainMasteryOverall = avg(Object.values(domainScores));

        // Skill gap index: ratio of repeating weak skills (higher => worse)
        const totalWeakSkillUnique = Object.keys(weakSkillOccurrences).length;
        let repeatedWeakCount = 0;
        Object.values(weakSkillOccurrences).forEach(c => { if (c > 1) repeatedWeakCount += (c - 1); });
        // scale 0..1 where 0 = no repeated weak skills, 1 = many repeats relative to sessions
        const skillGapIndex = totalSessions ? Math.min(1, repeatedWeakCount / Math.max(1, totalSessions)) : 0;

        // Consistency score: activeDays / daysBetween(firstSession and today)
        const daysActive = activeDaysSet.size;
        const daysSince = firstSessionDate ? daysBetween(firstSessionDate, new Date()) + 1 : 1;
        const consistencyScore = +(daysSince > 0 ? (daysActive / daysSince) : 0).toFixed(3);

        // Engagement score (normalized)
        const engagementScore = engagementCount;

        // Skill diversity
        const skillDiversity = profileSkills.length;

        // Accuracy score = average session accuracy (0..100 scale expected)
        const accuracyScore = +(avg(accuracyList) || 0).toFixed(3);

        // Growth / progression: compare first session avg vs last session avg
        const firstAvg = sessionAverages.length ? sessionAverages[0] : 0;
        const lastAvg = sessionAverages.length ? sessionAverages[sessionAverages.length - 1] : 0;
        const growthScore = +(lastAvg - firstAvg).toFixed(3);

        // Difficulty progression: if later sessions have higher avg than earlier -> positive
        const half = Math.ceil(sessionAverages.length / 2);
        const earlyAvg = avg(sessionAverages.slice(0, half));
        const lateAvg = avg(sessionAverages.slice(half));
        const difficultyProgression = +(lateAvg - earlyAvg).toFixed(3);

        // Failure recovery: fraction of sessions where average increased compared to previous
        let recoveryCount = 0;
        for (let i = 1; i < sessionAverages.length; i++) {
            if (sessionAverages[i] > sessionAverages[i - 1]) recoveryCount++;
        }
        const failureRecoveryRatio = sessionAverages.length > 1 ? +(recoveryCount / (sessionAverages.length - 1)).toFixed(3) : 0;

        // Learning path coverage for targetDomain (simple heuristic: sessions in target / 5)
        const targetSessionsCount = targetDomain ? (sessions.filter(s => s.domain === targetDomain).length) : 0;
        const learningPathCoverage = Math.min(100, Math.round((targetSessionsCount / 5) * 100)); // 5 sessions = 100% coverage

        // XP calculation (composite: many contributors)
        let XP = BASE_XP;
        XP += totalSessions * XP_WEIGHTS.PER_SESSION;
        XP += totalQuestions * XP_WEIGHTS.PER_QUESTION;
        XP += totalStrongSkills * XP_WEIGHTS.PER_STRONG_SKILL;
        XP += domainMasteryOverall * XP_WEIGHTS.DOMAIN_MASTERY_MULTIPLIER;
        XP += consistencyScore * XP_WEIGHTS.CONSISTENCY_MULTIPLIER;
        XP += engagementScore * XP_WEIGHTS.ENGAGEMENT_MULTIPLIER;
        XP += Math.max(0, growthScore) * XP_WEIGHTS.GROWTH_MULTIPLIER; // reward positive growth only
        XP += accuracyScore * XP_WEIGHTS.ACCURACY_MULTIPLIER;

        // Round XP
        XP = Math.round(XP);

        const level = calcLevelFromXP(XP);

        // Auto badges rules (comprehensive)
        const badges = new Set();

        // Basic milestones
        if (totalSessions >= 1) badges.add("🎖️ First Step");
        if (totalSessions >= 5) badges.add("🏁 Active Challenger (5 sessions)");
        if (totalSessions >= 10) badges.add("🏆 Veteran Challenger (10 sessions)");

        // Questions
        if (totalQuestions >= 1) badges.add("🧠 Curious Mind");
        if (totalQuestions >= 10) badges.add("📚 Inquisitive (10+ questions)");
        if (totalQuestions >= 50) badges.add("📖 Question Master (50+)");

        // Domains
        if (domainMap.size >= 2) badges.add("🌐 Multi-Domain Explorer");
        if (domainMap.size >= 5) badges.add("🌎 Polyglot Learner");

        // Skills
        if (strongSkillSet.size >= 3) badges.add("📈 Strong Skill Profile");
        if (skillDiversity >= 10) badges.add("📚 Multi-Skill Expert");
        if (skillDiversity >= 20) badges.add("🧬 Full-Stack Polymath");

        // Engagement & attempts
        if (engagementScore >= 1) badges.add("⚡ Active Participant");
        if (engagementScore >= 10) badges.add("💡 Engaged Learner");
        if (engagementScore >= 30) badges.add("🔥 Power Learner");

        // Attempts types
        sessionsSorted.forEach(s => {
            if (s?.detailed_summary?.session_stats?.coding_attempted > 0) badges.add("💻 Code Attempted");
            if (s?.detailed_summary?.session_stats?.voice_attempted > 0) badges.add("🎤 Voice Attempted");
        });

        // Accuracy and growth
        if (accuracyScore >= 60) badges.add("🎯 Accurate Performer");
        if (growthScore > 0.5) badges.add("📈 Rapid Improver");
        if (failureRecoveryRatio >= 0.5) badges.add("🔁 Resilient Learner");

        // Consistency
        if (consistencyScore >= 0.5) badges.add("📅 Consistent Learner");
        if (daysActive >= 3) badges.add("🔥 3-day Streak");
        if (daysActive >= 7) badges.add("🔥 7-day Streak");

        // Level-based
        if (level >= 2) badges.add(`🚀 Level ${level} Achiever`);
        if (XP >= 500) badges.add("🌟 Elite Contributor");

        // Learning path
        if (learningPathCoverage >= 20) badges.add("🎯 Learning Path Started");
        if (learningPathCoverage >= 100) badges.add("🏅 Learning Path Completed");

        // Skill gap warning badge (negative style)
        if (skillGapIndex > 0.5) badges.add("⚠️ Repeating Weaknesses");

        // Compose badge array sorted (nice)
        const badgesList = Array.from(badges);

        // Build final response object
        const response = {
            success: true,
            userId: user._id,
            meta: {
                firstSessionAt: firstSessionDate ? firstSessionDate.toISOString() : null,
                lastSessionAt: lastSessionDate ? lastSessionDate.toISOString() : null,
                daysActive,
                daysSinceFirstSession: daysSince
            },
            overall: {
                totalSessions,
                totalQuestions,
                totalStrongSkills,
                totalWeakSkills,
                domainCount: domainMap.size,
                skillDiversity
            },
            domainMastery: domainScores,
            scores: {
                XP,
                level,
                domainMasteryOverall: +domainMasteryOverall.toFixed(3),
                skillGapIndex: +skillGapIndex.toFixed(3),
                consistencyScore,
                engagementScore,
                accuracyScore,
                growthScore,
                difficultyProgression,
                failureRecoveryRatio,
                learningPathCoverage: learningPathCoverage // percentage
            },
            badges: badgesList
        };

        return res.json(response);

    } catch (err) {
        console.error("Advanced gamification error:", err);
        return res.status(500).json({ success: false, message: err.message });
    }
};