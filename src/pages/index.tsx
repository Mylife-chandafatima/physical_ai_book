import React from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";

import styles from "./index.module.css";

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={clsx("hero hero--primary", styles.heroBanner)}>
      <div className={clsx("container", styles.heroGrid)}>
        {/* Left side → text + buttons + badges */}
        <div className={styles.heroContent}>
          <h1 className="hero__title">{siteConfig.title}</h1>
          <p className="hero__subtitle">{siteConfig.tagline}</p>
          <p className={styles.authorCredit}>By Chanda Fatima</p>

          <div className={styles.buttons}>
            <Link
              className="button button--secondary button--lg"
              to="/docs/module-1/introduction"
            >
              Start Reading 📚
            </Link>
            <Link
              className="button button--outline button--lg"
              to="https://github.com/Mylife-chandafatima"
            >
              View on GitHub 🌟
            </Link>
          </div>

          <div className={styles.badges}>
            <span className={styles.badge}>Open Source</span>
            <span className={styles.badge}>Physical AI</span>
            <span className={styles.badge}>Comprehensive</span>
          </div>
        </div>

        {/* Right side → robot image */}
        <div className={styles.robotImageContainer}>
          <img
            src="/img/Image2.jpeg"
            alt="Humanoid Robot"
            className={styles.robotImage}
          />
        </div>
      </div>
    </header>
  );
}

// Learning Spectrum data
const learningSpectrum = [
  {
    stage: "Foundations",
    title: "Robotic Nervous System",
    description:
      "Master ROS 2 fundamentals, nodes, topics, services, and URDF modeling",
    icon: "🧩",
    docId: "module-1/introduction",
  },
  {
    stage: "Intelligent Systems",
    title: "Digital Twin & AI Brain",
    description:
      "Build high-fidelity simulations and implement AI perception systems",
    icon: "🧠",
    docId: "module-2/introduction",
  },
  {
    stage: "Advanced AI",
    title: "Vision-Language-Action",
    description:
      "Create robots that understand and execute complex natural language commands",
    icon: "🚀",
    docId: "module-4/introduction-to-vla-concepts",
  },
];

// Core pillars data
const corePillars = [
  {
    title: "ROS 2 Integration",
    description:
      "Learn to build robust robotic systems with Robot Operating System 2",
    icon: "🤖",
    docId: "module-1/introduction",
  },
  {
    title: "Digital Twins",
    description:
      "Master simulation environments using Gazebo and Unity for safe development",
    icon: "🌐",
    docId: "module-2/introduction",
  },
  {
    title: "NVIDIA Isaac",
    description: "Implement AI-powered robotics with NVIDIA Isaac platform",
    icon: "🧠",
    docId: "module-3/introduction",
  },
  {
    title: "Vision-Language-Action",
    description:
      "Create systems that understand and act on natural language commands",
    icon: "👁️",
    docId: "module-4/introduction-to-vla-concepts",
  },
  {
    title: "Embodied Intelligence",
    description:
      "Develop AI that learns through physical interaction with the environment",
    icon: "💡",
    docId: "module-4/cognitive-planning-llms",
  },
  {
    title: "Sim-to-Real Transfer",
    description: "Bridge the gap between simulation and real-world deployment",
    icon: "🔄",
    docId: "module-3/sim-to-real-transfer",
  },
];

// Learning Journey data
const learningJourney = [
  {
    level: 1,
    title: "ROS 2 Fundamentals",
    subtitle: "Robotic Nervous System",
    description:
      "Learn the basics of ROS 2 architecture, nodes, topics, and services",
    approach: "Hands-on coding with practical examples",
    outcome: "Build your first ROS 2 package",
    docId: "module-1/introduction",
  },
  {
    level: 2,
    title: "Simulation & Modeling",
    subtitle: "Digital Twin Environments",
    description: "Create realistic simulations using Gazebo and Unity",
    approach: "Physics-based modeling and sensor simulation",
    outcome: "Deploy a complete simulation environment",
    docId: "module-2/introduction",
  },
  {
    level: 3,
    title: "AI Integration",
    subtitle: "NVIDIA Isaac Platform",
    description: "Implement perception and navigation systems with Isaac",
    approach: "Deep learning and computer vision integration",
    outcome: "Build an autonomous navigation system",
    docId: "module-3/introduction",
  },
  {
    level: 4,
    title: "Advanced Control",
    subtitle: "Vision-Language-Action Systems",
    description: "Create robots that respond to natural language commands",
    approach: "Multimodal AI and cognitive planning",
    outcome: "Deploy a VLA system for complex tasks",
    docId: "module-4/introduction-to-vla-concepts",
  },
  {
    level: 5,
    title: "Research & Deployment",
    subtitle: "Real-World Applications",
    description: "Bridge sim-to-real gap and deploy in real environments",
    approach: "Advanced transfer learning and optimization",
    outcome: "Complete humanoid robot deployment",
    docId: "module-4/capstone-example",
  },
];

// The Great Shift comparison
const theGreatShift = {
  traditional: [
    "Pre-programmed behaviors",
    "Limited adaptability",
    "Separate perception and action",
    "Rigid control systems",
    "Manual programming",
  ],
  physicalAI: [
    "Learning through interaction",
    "Adaptive to environment",
    "Integrated perception-action loops",
    "Flexible cognitive systems",
    "Autonomous learning",
  ],
};

function PillarCard({ icon, title, description, docId }) {
  return (
    <div className={clsx("col col--4")}>
      <Link to={`/docs/${docId}`} className={styles.pillarLink}>
        <div className="text--center padding-horiz--md">
          <div className={styles.pillarIcon}>{icon}</div>
          <h3>{title}</h3>
          <p>{description}</p>
          <div className={styles.readMore}>Read more →</div>
        </div>
      </Link>
    </div>
  );
}

function SpectrumStageCard({ stage, title, description, icon, docId, isMain }) {
  return (
    <div className={clsx("col col--4")}>
      <Link to={`/docs/${docId}`} className={styles.spectrumLink}>
        <div
          className={clsx(
            "card",
            styles.spectrumCard,
            isMain ? styles.mainStage : ""
          )}
        >
          <div className="card__header text--center">
            <div className={styles.stageIcon}>{icon}</div>
            <h3>{stage}</h3>
            <h4>{title}</h4>
          </div>
          <div className="card__body">
            <p>{description}</p>
            <div className={styles.readMore}>Explore →</div>
          </div>
        </div>
      </Link>
    </div>
  );
}

function JourneyStepCard({
  level,
  title,
  subtitle,
  description,
  approach,
  outcome,
  docId,
}) {
  return (
    <div className={clsx("col col--2")}>
      <Link to={`/docs/${docId}`} className={styles.journeyLink}>
        <div className={clsx("card", styles.journeyCard)}>
          <div className="card__header text--center">
            <div className={styles.levelBadge}>Level {level}</div>
            <h3>{title}</h3>
            <small>{subtitle}</small>
          </div>
          <div className="card__body">
            <p>{description}</p>
            <div className={styles.journeyDetails}>
              <div>
                <strong>Approach:</strong> {approach}
              </div>
              <div>
                <strong>Outcome:</strong> {outcome}
              </div>
            </div>
            <div className={styles.readMore}>Learn →</div>
          </div>
        </div>
      </Link>
    </div>
  );
}

function ShiftCard({ title, items }) {
  return (
    <div className="col col--6">
      <div className={clsx("card", styles.shiftCard)}>
        <div className="card__header text--center">
          <h3>{title}</h3>
        </div>
        <div className="card__body">
          <ul>
            {items.map((item, idx) => (
              <li key={idx}>{item}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        {/* Learning Spectrum Section */}
        <div className="text--center padding-vert--md">
          <h1 className={styles.mainTitle}>Learning Spectrum</h1>
          <p className="padding-horiz--lg">
            Progress from foundational robotics to advanced Physical AI systems
          </p>
        </div>
        <div className="row">
          {learningSpectrum.map((stage, idx) => (
            <SpectrumStageCard
              key={idx}
              stage={stage.stage}
              title={stage.title}
              description={stage.description}
              icon={stage.icon}
              docId={stage.docId}
              isMain={idx === 1} // Highlight the middle stage
            />
          ))}
        </div>

        {/* Core Pillars Section */}
        <div className="text--center padding-vert--md">
          <h1 className={styles.mainTitle}>Core Pillars</h1>
          <p className="padding-horiz--lg">
            Foundational technologies that power modern humanoid robotics
          </p>
        </div>
        <div className="row">
          {corePillars.map((pillar, idx) => (
            <PillarCard
              key={idx}
              icon={pillar.icon}
              title={pillar.title}
              description={pillar.description}
              docId={pillar.docId}
            />
          ))}
        </div>

        {/* Learning Journey Section */}
        <div className="text--center padding-vert--md">
          <h1 className={styles.mainTitle}>Learning Journey</h1>
          <p className="padding-horiz--lg">
            Progress through 5 levels from fundamentals to advanced deployment
          </p>
        </div>
        <div className="row">
          {learningJourney.map((step, idx) => (
            <JourneyStepCard
              key={idx}
              level={step.level}
              title={step.title}
              subtitle={step.subtitle}
              description={step.description}
              approach={step.approach}
              outcome={step.outcome}
              docId={step.docId}
            />
          ))}
        </div>

        {/* The Great Shift Section */}
        <div className="text--center padding-vert--md">
          <h1 className={styles.mainTitle}>The Great Shift</h1>
          <p className="padding-horiz--lg">
            From Traditional Robotics to Physical AI & Embodied Intelligence
          </p>
        </div>
        <div className="row">
          <ShiftCard
            title="🤖 Traditional Robotics"
            items={theGreatShift.traditional}
          />
          <ShiftCard
            title="🧠 Physical AI & Embodied Intelligence"
            items={theGreatShift.physicalAI}
          />
        </div>

        {/* Call to Action */}
        <div className="text--center padding-vert--md">
          <div className="row">
            <div className="col col--12">
              <Link
                className="button button--primary button--lg"
                to="/docs/module-1/introduction"
              >
                Start Your Journey 🚀
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const { siteConfig } = useDocusaurusContext();

  return (
    <Layout
      title={`Home - ${siteConfig.title}`}
      description="A comprehensive guide to Physical AI and Humanoid Robotics by Chand Afatima"
    >
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
